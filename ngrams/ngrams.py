import pandas as pd
import numpy as np
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
            annotations_list1 = [token, label, end-start]    
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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label', 'ngram'])

        # Loop through all the annotators
        for annotator in self.annotator_list:
            # Get the annotator_id from the annotator object (assuming it has an 'id' attribute)
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)

            # Create a temporary DataFrame to store the current annotator's data
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'ngram'])
            temp_df['annotator_id'] = annotator_id

            # Append the temporary DataFrame to the main DataFrame using pandas.concat
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df
    
    def split_df_into_dfngrams(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}    
        # Loop through the dataframe df and create a dataframe for each ngram
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            # drop the ngram column
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        return ngrams_dfs

    def get_all_ngrams_agreements_lists(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}
        # Create a dictionary to store the ngrams agreements lists
        ngram_agreements = {}
        # Call split_df_into_dfngrams to create a dataframe for each ngram
        ngrams_dfs = self.split_df_into_dfngrams(df)
        # Loop through all the dataframes in ngrams 
        for ngram, ngram_df in ngrams_dfs.items():
            # Get the list of partial and full agreements for the current ngram
            ngram_agreements_list = self.get_single_ngram_agreement_list(ngram_df)
            # Add the list of partial and full agreements to the ngram_agreements dictionary
            ngram_agreements[f'{ngram}-ngram'] = ngram_agreements_list
        all_ngram_agreements = [{key: list(agreement)} for key, agreement in ngram_agreements.items()]
        return all_ngram_agreements 
    
    def get_single_ngram_agreement_list(self, df):         
        partial_agreements = 0
        full_agreements = 0
        
        for row in range(len(df.index)):
            row_values = df.iloc[row].values
            none_count = (row_values == 'None').sum()
            
            if none_count == 0 or none_count == len(row_values):
                full_agreements += 1
            else:
                partial_agreements += 1
        
        return full_agreements, partial_agreements

    def create_single_annotations_table(self, annotated_df):
        # Create the pivot table with 'token' as index and 'annotator_id' as columns
        pivot_df = annotated_df.pivot_table(index=['token', 'ngram'], columns='annotator_id', values='label', aggfunc='first')

        # Reset the index to make 'token' and 'ngram' regular columns
        pivot_df.reset_index(inplace=True)

        # Set the 'token' column as the index again
        result_df = pivot_df.set_index('token')
        result_df = result_df.fillna('None')
        return result_df

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

    def get_agreement_percentages(self, data):
        new_data = []

        for annotator_data in data:
            annotator = list(annotator_data.keys())[0]
            agreement, disagreement = annotator_data[annotator]
            
            if agreement + disagreement > 0:
                percentage_agreement = agreement / (agreement + disagreement)
            else:
                percentage_agreement = 0.0

            new_data.append({annotator: percentage_agreement})

        return new_data


    def create_agreement_summary(self, agreements_list):
        agreement_ranges = {
            "lowest agreement": (0, 20),
            "medium-low agreement": (20, 40),
            "medium agreement": (40, 60),
            "medium-high agreement": (60, 80),
            "high agreement": (80, 100)
        }

        annotators_dfs = {}

        for annotator_data in agreements_list:
            annotator = list(annotator_data.keys())[0]
            tokens_data = annotator_data[annotator]
            summary_data = {key: [] for key in agreement_ranges.keys()}

            for token_data in tokens_data:
                token = token_data[0]
                agreement_percentage = token_data[1] * 100

                for range_name, (low, high) in agreement_ranges.items():
                    if low <= agreement_percentage < high or agreement_percentage == 100 and range_name == "high agreement":
                        summary_data[range_name].append(token)
                        break  # Add this line to exit the loop once the token is appended

            annotators_dfs[annotator] = pd.DataFrame(dict([(k, pd.Series(v, dtype='object')) for k, v in summary_data.items()]))

        return annotators_dfs

    
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




                
                
                











    


