import pandas as pd
import numpy as np
import networkx as nx
from annotator import Annotator


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

    # Create a tripartite graph
    def create_tripartate_graph(self, data: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for index, row in data.iterrows():
            annotator = row['annotator_id']
            item = row['token']
            annotation = row['label']
            G.add_edge(annotator, item)
            G.add_edge(item, annotation)
        return G
    
    # Implement the FolkRank algorithm
    def get_folkrank(self, G, d=0.85, max_iter=100, tol=1e-6, preference_vector=None):
        nodes_list = list(G.nodes())
        n = len(nodes_list)
        node_idx = {node: idx for idx, node in enumerate(nodes_list)}

        A = nx.to_numpy_array(G)
        A = A / A.sum(axis=0, keepdims=True)  # Normalize columns
        P = (1 - d) * np.eye(n) + d * A

        # Initialize preference vector
        if preference_vector is None:
            p = np.ones(n) / n
        else:
            p = np.array([preference_vector.get(node, 0) for node in nodes_list])
            p = p / p.sum()

        # Power iteration
        for _ in range(max_iter):
            p_next = P @ p
            if np.linalg.norm(p_next - p, ord=1) < tol:
                break
            p = p_next

        # Get the final ranking
        ranking = {nodes_list[i]: score for i, score in enumerate(p)}

        return ranking 




    




                
                
                











    


