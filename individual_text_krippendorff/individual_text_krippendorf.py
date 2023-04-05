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
        self.labels = ['Item', 'Activity', 'Location', 'Time', 'Attribute', 'Cardinality', 'Agent', 'Consumable', 'Observation/Observed_state', 'Observation/Quantitative', 'Observation/Qualitative', 'Specifier', 'Event', 'Unsure', 'Typo', 'Abbreviation']

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
 
    def get_tokens_coefficients(self, df) -> float:
        """
            Calculate Coefficients for all documents

            Returns:
                The Coefficient value for all documents
        """
        tokens_dict = {}    
        # iterate through df rows and remove all rows except the current row
        for index, row in df.iterrows():
            df1 = df.drop(index)
            # calculate the krippendorff coefficient for the current row
            cac_coefficient = CAC(df1)
            krippendorff_values = cac_coefficient.krippendorff()
            krippendorff_alpha = krippendorff_values['est']['coefficient_value']
            # if index already exists in the dictionary, append the value to the list
            if index in tokens_dict:
                tokens_dict[index].append(krippendorff_alpha)
            else :
                tokens_dict[index] = krippendorff_alpha
        return tokens_dict

    def get_average_tokens_coefficients(self, tokens_dict: dict) -> dict:
        """
            Calculate the average coefficient value for each token

            Returns:
                The average coefficient value for each token
        """
        # Initialize a dictionary to store the average coefficient values for each token
        average_coefficients_dict = {}
        # loop through tokens_dict and calculate the average coefficient value for each token
        for key, value in tokens_dict.items():
            average_coefficients_dict[key] = sum(value)/len(value)
        return average_coefficients_dict
    
    def insert_token_coefficients_to_df(self, average_coefficients_dict: dict) -> pd.DataFrame:
        """
            Create a new DataFrame and insert the average coefficient value for each token into the DataFrame

            Returns:
                The DataFrame with the average coefficient value for each token
        """
        # create a new DataFrame
        df = pd.DataFrame(columns=['token', 'percent_agreement'])
        # loop through the dictionary and insert the average coefficient value for each token into the new DataFrame
        for key, value in average_coefficients_dict.items():
            df = df.append({'token': key, 'percent_agreement': value}, ignore_index=True)
        return df

    def create_tokens_percentage_agreement_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Create a new DataFrame with the average coefficient value for each token

            Returns:
                The DataFrame with the average coefficient value for each token
        """
        # Create a new DataFrame with the average coefficient value for each token
        tokens_dict = self.get_tokens_coefficients(df)
        average_coefficients_dict = self.get_average_tokens_coefficients(tokens_dict)
        token_percentages_df = self.insert_token_coefficients_to_df(average_coefficients_dict)
        return token_percentages_df

    def create_agreement_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset the index to make 'token' a column
        df = df.reset_index()        
        # Create three empty lists for low, medium, and high agreement tokens
        lowest_agreement = []
        medium_low_agreement = []
        medium_agreement = []
        medium_high_agreement = []
        high_agreement = []

        # Iterate through the rows of the dataframe and append tokens to the corresponding list
        for index, row in df.iterrows():
            if 0 <= row['percent_agreement'] < 20:
                lowest_agreement.append(row['token'])
            elif 20 <= row['percent_agreement'] < 40:
                medium_low_agreement.append(row['token'])
            elif 40 <= row['percent_agreement'] < 60:
                medium_agreement.append(row['token'])
            elif 60 <= row['percent_agreement'] < 80:
                medium_high_agreement.append(row['token'])
            elif 80 <= row['percent_agreement'] <= 100:
                high_agreement.append(row['token'])

        # Create a new dataframe with the low, medium, and high agreement columns
        new_df = pd.DataFrame({
            'lowest agreement': pd.Series(lowest_agreement, dtype='object'),
            'medium low agreement': pd.Series(medium_low_agreement, dtype='object'),
            'medium agreement': pd.Series(medium_agreement, dtype='object'),
            'medium high agreement': pd.Series(medium_high_agreement, dtype='object'),
            'high agreement': pd.Series(high_agreement, dtype='object')
        })
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

    




                
                
                











    


