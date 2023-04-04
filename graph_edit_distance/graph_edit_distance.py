import pandas as pd
import numpy as np
import networkx as nx
from annotator import Annotator
import itertools
from networkx.algorithms import isomorphism
import grakel
from grakel.kernels import ShortestPath

class Label_Metrics :

    def __init__(self, *args):
        """
            Class for obtaining the annotator individual label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        self.annotator_list = list(args)
        self.annotator_count = len(self.annotator_list)
        self.annotator_numbered_list = []
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
            Gets all the same annotated document ids for all the annotators

            Returns:
                The same annotated document ids annotated by all the annotators 

        """ 
        # Initialize a set with the doc_idxs from the first annotator
        same_docs = set(self.annotator_list[0].get_doc_idxs())

        # Iterate through the remaining annotators, updating the set with the intersection
        for annotator in self.annotator_list[1:]:
            same_docs.intersection_update(annotator.get_doc_idxs())

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

    def get_all_annotators_tokens_labels_single_doc(self, doc_idx) -> pd.DataFrame:
        """
            Gets the tokens and labels for a doc_idx for all the annotators

            Parameters:
                doc_idx :
                    the document id

            Returns:
                The tokens with labels as a DataFrame for all the annotators

        """
        # Initialize an empty DataFrame with the desired columns
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label'])

        # Loop through all the annotators
        for annotator in self.annotator_list:
            # Get the annotator_id from the annotator object (assuming it has an 'id' attribute)
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)

            # Create a temporary DataFrame to store the current annotator's data
            temp_df = pd.DataFrame(annotated, columns=['token', 'label'])
            temp_df['annotator_id'] = annotator_id

            # Append the temporary DataFrame to the main DataFrame using pandas.concat
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df

    def create_single_annotations_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        # Pivot the annotated_df DataFrame to create a table with tokens as rows and annotators, labels as columns
        table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')

        # Ensure the table contains dtype 'object' and missing values are replaced with None
        table = table.astype(object).where(pd.notnull(table), None)
        return table

    def get_accumulated_table(self) -> pd.DataFrame:
        """
            Get the accumulated table for all the documents

            Returns:
                The accumulated table for all the documents
        """
        # Get the same document ids annotated by all the annotators
        same_docs = self.get_same_doc_ids()

        # Initialize an empty dataframe to accumulate all subdocuments' annotations
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            # Get the tokens and labels for a doc_idx for all the annotators
            annotated_df = self.get_all_annotators_tokens_labels_single_doc(doc_idx)

            # Create a table with tokens as rows and annotators as columns
            table = self.create_single_annotations_table(annotated_df)

            # Accumulate the annotations in the accumulated_coefficients_table
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table
 
    def pivot_dataframe(self, pivot_table: pd.DataFrame) -> pd.DataFrame:
        # Reset the index of the pivot_table to bring 'token' back as a column
        pivot_table_reset = pivot_table.reset_index()
        long_format_df = pd.melt(pivot_table_reset, id_vars='token', var_name='annotator_id', value_name='label')
        long_format_df = long_format_df[['annotator_id', 'token', 'label']]
        long_format_df = long_format_df.dropna(subset=['label'])
        return long_format_df

    def create_all_annotations_table(self) -> float:
        """
            Create a table with annotators as rows and token, annotation as columns for the whole corpus

            Returns:
                The dataframe with annotators as rows and token, annotation as columns for the whole corpus
        """
        # Get the same document ids annotated by all the annotators
        same_docs = self.get_same_doc_ids()
     
        # Initialize an empty dataframe to accumulate all subdocuments' annotations
        accumulated_table = pd.DataFrame()

        for doc_idx in same_docs:
            # Get the tokens and labels for a doc_idx for all the annotators
            table = self.get_all_annotators_tokens_labels_single_doc(doc_idx)

            # Accumulate the annotations in the accumulated_coefficients_table
            accumulated_table = pd.concat([accumulated_table, table], axis=0)

        return accumulated_table

    @classmethod
    def create_annotator_graph(cls,data: pd.DataFrame, annotator: str) -> nx.Graph:
        G = nx.Graph()
        annotator_data = data[data['annotator_id'] == annotator]
        for index, row in annotator_data.iterrows():
            item = row['token']
            annotation = row['label']
            G.add_edge(item, annotation)
        return G

    @classmethod
    def create_annotator_graphs(cls, data: pd.DataFrame, annotator_nodes: set) -> dict:
        return {annotator: cls.create_annotator_graph(data, annotator) for annotator in annotator_nodes}

    @classmethod
    def calculate_pairwise_ged(cls, annotator_graphs: dict, annotator_nodes: set) -> dict:
        pairwise_ged = {}
        for annotator1, annotator2 in itertools.combinations(annotator_nodes, 2):
            pair_key = (annotator1, annotator2)
            if pair_key in pairwise_ged:
                continue
            ged_value = nx.graph_edit_distance(annotator_graphs[annotator1], annotator_graphs[annotator2], timeout = 20)
            pairwise_ged[pair_key] = ged_value
        return pairwise_ged
  
    @staticmethod
    def calculate_pairwise_reliability(pairwise_ged: dict, annotator_graphs: dict) -> dict:
        pairwise_reliability = {}
        for pair, ged in pairwise_ged.items():
            annotator1, annotator2 = pair
            graph1 = annotator_graphs[annotator1]
            graph2 = annotator_graphs[annotator2]

            total_nodes = len(graph1.nodes) + len(graph2.nodes)
            reliability = 1 - (ged / total_nodes)
            pairwise_reliability[pair] = reliability
        return pairwise_reliability

    @staticmethod
    def calculate_overall_reliability(pairwise_reliability: dict) -> dict:
        overall_reliability = sum(pairwise_reliability.values()) / len(pairwise_reliability)
        return {'Overall Reliability' : overall_reliability}
    



    @classmethod
    def calculate_pairwise_mcs(cls, annotator_graphs: dict, annotator_nodes: set) -> dict:
        pairwise_mcs = {}
        for annotator1, annotator2 in itertools.combinations(annotator_nodes, 2):
            pair_key = (annotator1, annotator2)
            if pair_key in pairwise_mcs:
                continue

            GM = isomorphism.GraphMatcher(annotator_graphs[annotator1], annotator_graphs[annotator2])
            mcs_size = max([len(GM.subgraph_is_isomorphic()) for GM in GM.subgraph_isomorphisms_iter()])

            pairwise_mcs[pair_key] = mcs_size
        return pairwise_mcs

    @classmethod
    def calculate_pairwise_jaccard_similarity(cls, annotator_graphs: dict, annotator_nodes: set) -> dict:
        pairwise_jaccard_similarity = {}
        for annotator1, annotator2 in itertools.combinations(annotator_nodes, 2):
            pair_key = (annotator1, annotator2)
            if pair_key in pairwise_jaccard_similarity:
                continue
            nodes_annotator1 = set(annotator_graphs[annotator1].nodes())
            nodes_annotator2 = set(annotator_graphs[annotator2].nodes())
            intersection = nodes_annotator1.intersection(nodes_annotator2)
            union = nodes_annotator1.union(nodes_annotator2)
            jaccard_similarity = len(intersection) / len(union)
            pairwise_jaccard_similarity[pair_key] = jaccard_similarity
        return pairwise_jaccard_similarity

    @classmethod
    def calculate_pairwise_shortest_path_kernel(cls, annotator_graphs: dict, annotator_nodes: set) -> dict:
        pairwise_shortest_path_kernel = {}
        kernel = ShortestPath(normalize=True)

        # Convert NetworkX graphs to GraKeL graphs
        grakel_graphs = {annotator: grakel.graph_from_networkx(annotator_graphs[annotator], node_labels_type=str, edge_labels_type=str) for annotator in annotator_nodes}

        # Compute the shortest path kernel
        kernel_matrix = kernel.fit_transform(list(grakel_graphs.values()))

        # Store the pairwise kernel values in a dictionary
        for i, annotator1 in enumerate(annotator_nodes):
            for j, annotator2 in enumerate(annotator_nodes):
                if i < j:
                    pair_key = (annotator1, annotator2)
                    pairwise_shortest_path_kernel[pair_key] = kernel_matrix[i, j]

        return pairwise_shortest_path_kernel
    




                
                
                











    


