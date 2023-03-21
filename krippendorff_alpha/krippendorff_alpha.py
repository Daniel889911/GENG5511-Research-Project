import pandas as pd
import numpy as np
import krippendorff
from annotator import Annotator
from statistics import multimode

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

    def get_same_doc_ids(self) : 
        """
            Gets all the same annotated document ids for all the annotators

            Returns:
                The same annotated document ids annotated by all the annotators 
              
        """ 
        # find shortest document
        shortest_doc_length = float('inf')    
        shortest_doc_id = 0
        shortest_annotator = None
        doc_idxs1 = []
        doc_idxs2 = []
        for i, annotator in enumerate(self.annotator_list):
            doc_length = len(annotator.get_doc_idxs())
            if doc_length < shortest_doc_length:
                shortest_doc_length = doc_length
                shortest_annotator = self.annotator_list[i]
                shortest_doc_id = i
        doc_idxs1 = shortest_annotator.get_doc_idxs()
        self.same_docs = set(doc_idxs1)
        # compare the shortest document with all the other documents to get same documents
        for i, annotator in enumerate(self.annotator_list):            
            if shortest_doc_id != i:
                doc_idxs2 = annotator.get_doc_idxs()
                self.same_docs = self.same_docs.intersection(doc_idxs2)
        return list(self.same_docs)
    
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
            annotations_list1 = [token, label, start, end]    
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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label', 'start', 'end'])

        # Loop through all the annotators
        for annotator in self.annotator_list:
            # Get the annotator_id from the annotator object (assuming it has an 'id' attribute)
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)

            # Create a temporary DataFrame to store the current annotator's data
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'start', 'end'])
            temp_df['annotator_id'] = annotator_id

            # Append the temporary DataFrame to the main DataFrame using pandas.concat
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df


    def create_krippendorf_alpha_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        # Pivot the annotated_df DataFrame to create a table with annotators as rows and tokens as columns
        krippendorff_alpha_table = annotated_df.pivot_table(index='annotator_id', columns='token', values='label', aggfunc='first')

        # Ensure the table contains dtype 'object' and missing values are replaced with None
        krippendorff_alpha_table = krippendorff_alpha_table.astype(object).where(pd.notnull(krippendorff_alpha_table), None)

        return krippendorff_alpha_table

    def convert_krippendorf_alpha_table_to_list_of_lists(self, krippendorff_alpha_table: pd.DataFrame) -> list:
        """
            Convert a DataFrame to a list of lists

            Parameters:
                df :
                    The input DataFrame

            Returns:
                A list of lists representation of the input DataFrame
        """
        return krippendorff_alpha_table.values.tolist()

    def convert_labels_to_integers(self, data_list: list) -> list:
        """
            Convert labels in a list of lists to integers and replace None with NaN

            Parameters:
                data_list :
                    The input list of lists

            Returns:
                A list of lists with labels converted to integers and None replaced with NaN
        """
        # Get unique labels from the data_list (excluding None)
        unique_labels = set(x for sublist in data_list for x in sublist if x is not None)

        # Create a mapping of labels to integers
        label_to_int = {label: idx for idx, label in enumerate(unique_labels, 1)}

        # Apply the mapping to the list of lists
        converted_data_list = [[label_to_int.get(x, np.nan) if x is not None else np.nan for x in sublist] for sublist in data_list]

        return converted_data_list

    def calculate_krippendorf_alpha_for_all_docs(self) -> float:
        """
            Calculate Fleiss Kappa for all documents

            Returns:
                The Fleiss Kappa value for all documents
        """
        # Get the same document ids annotated by all the annotators
        same_docs = self.get_same_doc_ids()

        # Initialize a list to store the Krippendorff's alpha values for each document
        fleiss_kappa_values = []

        # Loop through all the documents
        for doc_idx in same_docs:
            # Get the tokens and labels for a doc_idx for all the annotators
            annotated_df = self.get_all_annotators_tokens_labels_single_doc(doc_idx)

            # Create a table with annotators as rows and tokens as columns
            fleiss_kappa_table = self.create_krippendorf_alpha_table(annotated_df)

            # Convert the table to a list of lists
            data_list = self.convert_krippendorf_alpha_table_to_list_of_lists(fleiss_kappa_table)

            # Convert labels to integers and replace None with NaN
            converted_data_list = self.convert_labels_to_integers(data_list)

            # Calculate Krippendorff's alpha
            fleiss_kappa = krippendorff.alpha(converted_data_list, level_of_measurement='nominal')

            # Add the Krippendorff's alpha value to the list
            fleiss_kappa_values.append(fleiss_kappa)

        # Calculate the mean Krippendorff's alpha value
        mean_fleiss_kappa = np.mean(fleiss_kappa_values)

        return mean_fleiss_kappa

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


                
                
                











    


