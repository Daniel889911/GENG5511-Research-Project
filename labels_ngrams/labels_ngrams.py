import pandas as pd
import numpy as np
from annotator import Annotator

class Labels_Ngram_Metrics :
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
    
    def get_token_label_ngrams(self, tokens:list, mentions: dict) -> list:
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

    def get_all_annotators_tokens_labels_single_doc_ngrams(self, doc_idx) -> pd.DataFrame:
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
            annotated = self.get_token_label_ngrams(token, mention)

            # Create a temporary DataFrame to store the current annotator's data
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'ngram'])
            temp_df['annotator_id'] = annotator_id

            # Append the temporary DataFrame to the main DataFrame using pandas.concat
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df

    def create_single_annotations_table_ngrams(self, annotated_df):
        # Create the pivot table with 'token' as index and 'annotator_id' as columns
        pivot_df = annotated_df.pivot_table(index=['token', 'ngram'], columns='annotator_id', values='label', aggfunc='first')

        # Reset the index to make 'token' and 'ngram' regular columns
        pivot_df.reset_index(inplace=True)

        # Set the 'token' column as the index again
        result_df = pivot_df.set_index('token')
        result_df = result_df.fillna('None')

        return result_df

    def get_accumulated_table_ngrams(self) -> pd.DataFrame:
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
            annotated_df = self.get_all_annotators_tokens_labels_single_doc_ngrams(doc_idx)

            # Create a table with tokens as rows and annotators as columns
            table = self.create_single_annotations_table_ngrams(annotated_df)

            # Accumulate the annotations in the accumulated_coefficients_table
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table

    def get_list_labels_for_ngrams(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}
        ngram_labels = {}
        # Loop through the dataframe df and create a dataframe for each ngram
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            # drop the ngram column
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        # Loop through all the dataframes in ngrams 
        for ngram, ngram_df in ngrams_dfs.items():
            # iterate through each column of the new table
            for col in ngrams_dfs[ngram].columns:
                unique_labels = []
                # find all unique labels in column
                unique_labels = ngrams_dfs[ngram][col].unique()
                # update ngram_labels[ngram] with the unique labels
                # ngram_labels[ngram] should only be updated with unique labels
                # if ngram_labels[ngram] is empty
                if ngram_labels.get(ngram) is None:
                    ngram_labels[ngram] = unique_labels.tolist()
                else:
                    for label in unique_labels:
                        if label not in ngram_labels[ngram]:
                            ngram_labels[ngram].append(label)
        return ngram_labels
    

    def get_ngrams_labels_agreements_lists(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}
        ngram_labels_agreements = {}
        # Loop through the dataframe df and create a dataframe for each ngram
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            # drop the ngram column
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        # Loop through all the dataframes in ngrams 
        for ngram, ngram_df in ngrams_dfs.items():
            # iterate through each column of the new table
            for col in ngrams_dfs[ngram].columns:
                unique_labels = []
                # find all unique labels in column
                unique_labels = ngrams_dfs[ngram][col].unique()
                # update ngram_labels[ngram] with the unique labels
                # ngram_labels[ngram] should only be updated with unique labels
                # if ngram_labels[ngram] is empty
                if ngram_labels_agreements.get(ngram) is None:
                    ngram_labels_agreements[ngram] = unique_labels.tolist()
                else:
                    for label in unique_labels:
                        if label not in ngram_labels_agreements[ngram]:
                            ngram_labels_agreements[ngram].append(label)
        return ngram_labels_agreements

    def get_all_ngrams_agreements_lists(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}    
        # Create a dictionary to store the ngrams agreements lists
        # The keys will be the ngrams and the values will be the lists of partial and full agreements
        # The lists will be in the form [full_agreements, partial_agreements]
        ngram_agreements = {}   
        # Loop through the dataframe df and create a dataframe for each ngram
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            # drop the ngram column
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        # Loop through all the dataframes in ngrams 
        for ngram, ngram_df in ngrams_dfs.items():
            # Get the list of partial and full agreements for the current ngram
            ngram_agreements_list = self.get_single_ngram_agreement_list(ngram_df)
            # Add the list of partial and full agreements to the ngram_agreements dictionary
            ngram_agreements[ngram] = ngram_agreements_list
        # Get the list of all the keys in the ngram_agreements dictionary
        all_keys = list(ngram_agreements.keys())
        all_ngram_agreements = [[key, ngram_agreements.get(key, 0)] for key in all_keys]     
        return ngram_agreements
    
    def get_single_ngram_agreement_list(self, df):         
        partial_agreements = 0
        full_agreements = 0
        for row in range(len(df.index)):
            if 'None' in df.iloc[row].values:
                partial_agreements += 1
            else:
                full_agreements += 1
        return [full_agreements, partial_agreements]

    def get_token_label_labels(self, tokens:list, mentions: dict) -> list:
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

    def get_all_annotators_tokens_labels_single_doc_labels(self, doc_idx) -> pd.DataFrame:
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
            annotated = self.get_token_label_labels(token, mention)

            # Create a temporary DataFrame to store the current annotator's data
            temp_df = pd.DataFrame(annotated, columns=['token', 'label'])
            temp_df['annotator_id'] = annotator_id

            # Append the temporary DataFrame to the main DataFrame using pandas.concat
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)

        return annotated_df

    def create_single_annotations_table_labels(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        # Pivot the annotated_df DataFrame to create a table with tokens as rows and annotators, labels as columns
        table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')

        # Ensure the table contains dtype 'object' and missing values are replaced with None
        table = table.astype(object).where(pd.notnull(table), None)
        return table

    def get_accumulated_table_labels(self) -> pd.DataFrame:
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
            annotated_df = self.get_all_annotators_tokens_labels_single_doc_labels(doc_idx)

            # Create a table with tokens as rows and annotators as columns
            table = self.create_single_annotations_table_labels(annotated_df)

            # Accumulate the annotations in the accumulated_coefficients_table
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table

    def get_label_agreements_lists(self, df):
        df = df.fillna('None')
        # Count the occurrences of each label for each token
        label_counts = df.apply(lambda x: x.value_counts(), axis=1)

        # Calculate the total number of annotators
        total_annotators = len(df.columns)

        # Initialize dictionaries to store the counts of partial and full agreements for each label
        partial_label_agreements = {}
        full_label_agreements = {}

        # Iterate through each token and its label counts
        for _, counts in label_counts.iterrows():
            # Check if there is a full agreement for any label
            full_agreement = any(count == total_annotators for count in counts)

            # If there is no full agreement, then it's a partial agreement
            if not full_agreement:
                for label, count in counts.items():
                    # get the highest count and label in counts.items()
                    if count == counts.max():
                        if not pd.isna(label) :
                            partial_label_agreements[label] = partial_label_agreements.get(label, 0) + 1
            else:
                # If there is a full agreement, increment the count for the fully agreed label
                majority_label = counts.idxmax()
                full_label_agreements[majority_label] = full_label_agreements.get(majority_label, 0) + 1

        all_keys = set(full_label_agreements.keys()) | set(partial_label_agreements.keys())
        all_label_agreements = [[key, full_label_agreements.get(key, 0), partial_label_agreements.get(key, 0)] for key in all_keys]

        return all_label_agreements       

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




                
                
                











    


