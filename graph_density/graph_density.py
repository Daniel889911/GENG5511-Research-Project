import pandas as pd
import numpy as np
import networkx as nx
from annotator import Annotator
import itertools
from networkx.algorithms import isomorphism

class Label_Metrics :

    def __init__(self, *args):
        """
            Class for obtaining the annotation individual label metrics 

            Parameters:
                annotations :
                    Instance of annotation class for a person
              
        """
        self.annotation_list = list(args)
        self.annotation_count = len(self.annotation_list)
        self.annotation_numbered_list = []
        self.same_docs = []
        self.annotated_corpus = []
        self.labels = ['Item', 'Activity', 'Location', 'Time', 'Attribute', 'Cardinality', 'Agent', 'Consumable', 'Observation/Observed_state', 'Observation/Quantitative', 'Observation/Qualitative', 'Specifier', 'Event', 'Unsure', 'Typo', 'Abbreviation']

    def list_To_String(self, List: list) -> str:
        """
            Converts a list into a string 

            Parameters:
                List :
                    The object of type list to convert to string
                    
            Returns:
                The converted object from list into type string
                
        """    
        str1 = " "    
        return (str1.join(List))

    def get_same_doc_ids(self): 
        """
            Gets all the same annotated document ids for all the annotations

            Returns:
                The same annotated document ids annotated by all the annotations 

        """ 
        # Initialize a set with the doc_idxs from the first annotation
        same_docs = set(self.annotation_list[0].get_doc_idxs())

        # Iterate through the remaining annotations, updating the set with the intersection
        for annotation in self.annotation_list[1:]:
            same_docs.intersection_update(annotation.get_doc_idxs())

        # Store the same annotated document ids in the class variable
        self.same_docs = same_docs

        return self.same_docs
    
    def get_token_label(self, tokens:list, mentions: dict) -> list:
        """
            Gets the combined token, labels, and gets the correct position of tokens

            Parameters:
                tokens :
                    The list of tokens for a document id
                mentions : 
                    The dictionary of mentions for a document id
                    
            Returns:
                The combined tokens with labels as a list for a document id 
              
        """
        annotations_list1 = []
        annotations_list2 = []
        for ment in mentions:
            start = ment["start"]
            end = ment["end"]
            token = tokens[start:end]
            token = self.list_To_String(token)
            label = ment["labels"]
            label = self.list_To_String(label)
            annotations_list1 = [token, label]    
            annotations_list2.append(annotations_list1)    
        return annotations_list2

    def get_all_annotations_tokens_labels_single_doc(self, doc_idx) -> pd.DataFrame:
        """
            Gets the tokens and labels for a doc_idx for all the annotations

            Parameters:
                doc_idx :
                    the document id

            Returns:
                The tokens with labels as a DataFrame for all the annotations

        """
        annotated_df = pd.DataFrame(columns=['annotation_id', 'token', 'label'])
        for annotation in self.annotation_list:
            annotation_id = annotation.name
            mention = annotation.get_doc_mentions(doc_idx)
            token = annotation.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)
            temp_df = pd.DataFrame(annotated, columns=['token', 'label'])
            temp_df['annotation_id'] = annotation_id
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df   

    def create_single_annotations_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        table = annotated_df.pivot_table(index='token', columns='annotation_id', values='label', aggfunc='first')
        table = table.astype(object).where(pd.notnull(table), None)
        return table

    def get_accumulated_table(self) -> pd.DataFrame:
        """
            Get the accumulated table for all the documents

            Returns:
                The accumulated table for all the documents
        """
        same_docs = self.get_same_doc_ids()
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            annotated_df = self.get_all_annotations_tokens_labels_single_doc(doc_idx)
            table = self.create_single_annotations_table(annotated_df)
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table
 
    def create_rows_same_labels(self, start_row: int, end_row: int, df: pd.DataFrame) -> pd.DataFrame:
        """
            Creates a new DataFrame with the same label for all the annotations

            Parameters:
                rows :
                    The number of rows to be same in the DataFrame
                df :
                    The DataFrame with the labels for all the annotations

            Returns:
                The new DataFrame with the same label for all the annotations
        """
        new_df = df.iloc[start_row:end_row].copy()
        for row_idx in range(start_row, end_row):
            row = df.iloc[row_idx]
            mode = row.mode()
            if not mode.empty:
                most_common_label = mode[0]
            else:
                most_common_label = self.labels[0]
            for col_idx in range(df.shape[1]):
                new_df.iloc[row_idx, col_idx] = most_common_label
        return new_df

    def create_rows_different_labels(self, start_row: int, end_row: int, df: pd.DataFrame) -> pd.DataFrame:
        """
            Creates a new DataFrame with different labels for all the annotations

            Parameters:
                rows :
                    The number of rows to be different in the DataFrame
                df :
                    The DataFrame with the labels for all the annotations

            Returns:
                The new DataFrame with different labels for all the annotations

        """
        new_df = df.iloc[start_row:end_row].copy()
        for row_idx in range(new_df.shape[0]):
            used_labels = []
            used_labels = [new_df.iloc[row_idx, 0]]
            available_labels = [label for label in self.labels if label not in used_labels]
            for col_idx in range(1, df.shape[1]):
                current_label = df.iloc[row_idx, col_idx]
                if current_label in available_labels:
                    new_label = current_label
                else:
                    new_label = available_labels.pop(0)
                new_df.iloc[row_idx, col_idx] = new_label
                used_labels.append(new_label)
                available_labels = [label for label in self.labels if label not in used_labels]
        return new_df
    
    def calculate_same_different_row_numbers(self, percentage: float, df: pd.DataFrame) -> tuple:
        total_rows = len(df)
        same_rows = int(total_rows * percentage)
        different_rows = total_rows - same_rows
        return (same_rows, different_rows, total_rows) 
 
    def create_df_same_different(self, percentage: float, df: pd.DataFrame) -> pd.DataFrame:
        same_rows, different_rows, total_rows = self.calculate_same_different_row_numbers(percentage, df)
        print(f'Creating a DataFrame with {same_rows} rows with same labels and {different_rows} rows with different labels')
        same_label_df = self.create_rows_same_labels(0, same_rows, df)
        different_label_df = self.create_rows_different_labels(same_rows, total_rows, df)
        result_df = pd.concat([same_label_df, different_label_df], axis=0)
        return result_df

    def pivot_dataframe(self, pivot_table: pd.DataFrame) -> pd.DataFrame:
        # Reset the index of the pivot_table to bring 'token' back as a column
        pivot_table_reset = pivot_table.reset_index()
        long_format_df = pd.melt(pivot_table_reset, id_vars='token', var_name='annotator', value_name='label')
        long_format_df = long_format_df[['annotator', 'token', 'label']]
        long_format_df = long_format_df.dropna(subset=['label'])
        return long_format_df

    def create_all_annotations_table(self) -> float:
        """
            Create a table with annotations as rows and token, annotation as columns for the whole corpus

            Returns:
                The dataframe with annotations as rows and token, annotation as columns for the whole corpus
        """
        same_docs = self.get_same_doc_ids()     
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            table = self.get_all_annotations_tokens_labels_single_doc(doc_idx)
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table

    @classmethod
    def create_agreement_graph(cls, df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in df.iterrows():
            annotator, token, label = row['annotator'], row['token'], row['label']
            G.add_node(annotator, type='annotation')
            G.add_node((token, label), type='annotation')
            G.add_edge(annotator, (token, label))
        return G

    @classmethod
    def custom_graph_density(cls, G):
        node_degrees = {}
        for node in G.nodes():
            if G.nodes[node]['type'] == 'annotation':
                node_degrees[node] = G.degree(node)
        
        total_annotation_edges = sum(node_degrees.values())
        
        # Count the total number of edges for annotation nodes with more than two edges
        edge_count = 0
        for node, degree in node_degrees.items():
            if degree > 2:
                edge_count += degree

        # Calculate the percentage of edges for nodes with more than one edge
        if total_annotation_edges == 0:
            return 0
        else:
            return edge_count / total_annotation_edges




    




                
                
                











    


