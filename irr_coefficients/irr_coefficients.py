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

        coefficients_dict = {}
      
        # Initialize an empty dataframe to accumulate all subdocuments' annotations
        accumulated_coefficients_table = pd.DataFrame()

        for doc_idx in same_docs:
            # Get the tokens and labels for a doc_idx for all the annotators
            annotated_df = self.get_all_annotators_tokens_labels_single_doc(doc_idx)

            # Create a table with annotators as rows and tokens as columns
            coefficients_table = self.create_annotations_table(annotated_df)

            # Accumulate the annotations in the accumulated_coefficients_table
            accumulated_coefficients_table = pd.concat([accumulated_coefficients_table, coefficients_table], axis=1)

        # Initialise CAC with the accumulated_coefficients_table
        cac_coefficient = CAC(accumulated_coefficients_table)

        # Calculate krippendorff coefficient value for the accumulated table
        krippendorff_values = cac_coefficient.krippendorff()
        krippendorff_alpha = krippendorff_values['est']['coefficient_value']

        # Calculate fleiss kappa coefficient value for the accumulated table
        fleiss_values = cac_coefficient.fleiss()
        fleiss_alpha =  fleiss_values['est']['coefficient_value']

        # Calculate Gwets AC1 value for the accumulated table
        gwet_values = cac_coefficient.gwet()
        gwet_alpha =  gwet_values['est']['coefficient_value']

        # Calculate Conger kappa value for the accumulated table
        conger_values = cac_coefficient.conger()
        conger_alpha =  conger_values['est']['coefficient_value']

        coefficients_dict['krippendorff'] = krippendorff_alpha
        coefficients_dict['fleiss'] = fleiss_alpha
        coefficients_dict['gwets'] = gwet_alpha
        coefficients_dict['conger'] = conger_alpha

        return coefficients_dict

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
    




                
                
                











    


