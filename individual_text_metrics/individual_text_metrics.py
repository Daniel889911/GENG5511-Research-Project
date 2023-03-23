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

    def get_all_annotators_tokens_labels_all_docs(self) -> pd.DataFrame:
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

        self.get_same_doc_ids()
    
        # Loop through all documents
        for doc_idx in self.same_docs:
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

    def create_individual_text_metrics_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        # Pivot the annotated_df DataFrame to create a table with tokens as rows and annotators as columns
        krippendorff_alpha_table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')

        # Replace missing values with np.nan
        krippendorff_alpha_table = krippendorff_alpha_table.replace(to_replace=[None], value=np.nan)

        # Calculate the mode of each row in the DataFrame
        mode_series = krippendorff_alpha_table.apply(lambda x: x.value_counts().idxmax() if x.value_counts().max() > 1 else 0, axis=1)

        # Calculate the percentage agreement for each row (token) in the DataFrame
        boolean_df = krippendorff_alpha_table.values == mode_series.values[:, np.newaxis]
        percentage_series = boolean_df.mean(axis=1) * 100   

        krippendorff_alpha_table['percent_agreement'] = percentage_series

        return krippendorff_alpha_table

    def calculate_individual_text_metrics_all_docs(self, df: pd.DataFrame) -> float:
        # Reset the index to make 'token' a column
        df = df.reset_index()

        # Create three empty lists for low, medium, and high agreement tokens
        low_agreement = []
        medium_agreement = []
        high_agreement = []

        # Iterate through the rows of the dataframe and append tokens to the corresponding list
        for index, row in df.iterrows():
            if row['percent_agreement'] < 30:
                low_agreement.append(row['token'])
            elif 30 <= row['percent_agreement'] <= 80:
                medium_agreement.append(row['token'])
            else:
                high_agreement.append(row['token'])

        # Create a new dataframe with the low, medium, and high agreement columns
        new_df = pd.DataFrame({'low_agreement': pd.Series(low_agreement), 'medium_agreement': pd.Series(medium_agreement), 'high_agreement': pd.Series(high_agreement)})
        new_df = new_df.replace(pd.NA, '')
        return new_df

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


                
                
                











    


