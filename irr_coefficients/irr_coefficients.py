import pandas as pd
import numpy as np
from irrCAC.raw import CAC
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


    def create_annotations_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        # Pivot the annotated_df DataFrame to create a table with annotators as rows and tokens as columns
        krippendorff_alpha_table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')

        # Ensure the table contains dtype 'object' and missing values are replaced with None
        krippendorff_alpha_table = krippendorff_alpha_table.astype(object).where(pd.notnull(krippendorff_alpha_table), None)

        return krippendorff_alpha_table

    def calculate_coefficient_for_all_docs(self) -> float:
        """
            Calculate Coefficients for all documents

            Returns:
                The Coefficient value for all documents
        """
        # Get the same document ids annotated by all the annotators
        same_docs = self.get_same_doc_ids()

        # Initialize a list to store the Krippendorff's alpha values for each document
        krippendorff_alpha_values_list = []
        fleiss_kappa_values_list = []
        gwets_values_list = []
        
        # Loop through all the documents
        for doc_idx in same_docs:
            # Get the tokens and labels for a doc_idx for all the annotators
            annotated_df = self.get_all_annotators_tokens_labels_single_doc(doc_idx)

            # Create a table with annotators as rows and tokens as columns
            coefficients_table = self.create_annotations_table(annotated_df)

            # Initialise CAC
            cac_coefficient = CAC(coefficients_table)

            # Calculate krippendorff coefficient value
            krippendorff_values = cac_coefficient.krippendorff()
            krippendorff_alpha = krippendorff_values['est']['coefficient_value']
            # Add the coefficient value to the list
            krippendorff_alpha_values_list.append(krippendorff_alpha)

            # Calculate fleiss coefficient value
            fleiss_kappa_values = cac_coefficient.fleiss()
            fleiss_kappa = fleiss_kappa_values['est']['coefficient_value']
            # Add the coefficient value to the list
            fleiss_kappa_values_list.append(fleiss_kappa)

            # Calculate gwets coefficient value
            gwets_values = cac_coefficient.gwet()
            gwets_ac1 = gwets_values['est']['coefficient_value']
            # Add the coefficient value to the list
            gwets_values_list.append(gwets_ac1)

        # Calculate the mean Krippendorff's alpha value
        krippendorf_mean_coefficient_value = np.mean(krippendorff_alpha_values_list)
        fleiss_kappa_mean_coefficient_value = np.mean(fleiss_kappa_values_list)
        gwets_mean_coefficient_value = np.mean(gwets_values_list)
        print(f"Krippendorff's alpha: {krippendorf_mean_coefficient_value}")
        print(f"Fleiss kappa: {fleiss_kappa_mean_coefficient_value}")
        print(f"Gwet's AC1: {gwets_mean_coefficient_value}")

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


                
                
                











    


