from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
from annotator import Annotator
import warnings

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
  
    def split_df_into_dfngrams(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}    
        # Loop through the dataframe df and create a dataframe for each ngram
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            # drop the ngram column
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        return ngrams_dfs
    
    def get_labels_and_ngrams(self, df):
        # Create a dictionary to store the ngrams dataframes
        ngrams_dfs = {}
        # Create a dictionary to store the ngrams metrics
        ngram_metrics = {}
        # Call split_df_into_dfngrams to create a dataframe for each ngram
        ngrams_dfs = self.split_df_into_dfngrams(df)
        # Loop through all the dataframes in ngrams 
        for ngram, ngram_df in ngrams_dfs.items():
            # Get the metrics for the current ngram
            label_metrics_list = self.get_label_agreements_with_chance_cohen(ngram_df)
            ngram_metrics_list = self.get_single_ngram_agreement_list(ngram_df)
            # Add the metrics to the ngram_metrics dictionary
            ngram_metrics[ngram] = label_metrics_list, ngram_metrics_list
        return ngram_metrics    
 
    def calculate_ngram_percentage_agreements(self, ngram_metrics_list):
        # Calculate the percentage agreement for each ngram metric
        ngram_percentage_metrics = ngram_metrics_list[0] / (ngram_metrics_list[0] + ngram_metrics_list[1])
        return ngram_percentage_metrics        
    
    def calculate_label_ngram_percentage_agreements(self, label_percentage_agreements, ngram_percentage_agreements, ngram):
        # Calculate the percentage agreement for each label/ngram metric
        label_ngram_percentage_metrics = {}
        for label, metrics in label_percentage_agreements.items():
            combined_metrics = (metrics + ngram_percentage_agreements)/2
            label_ngram_percentage_metrics[label] = ngram, combined_metrics
        return label_ngram_percentage_metrics
    
    def convert_labels_ngrams_metrics_for_heat_map(self, ngram_metrics):
        # Create a dictionary to store the ngrams dataframes
        heat_map_metrics = []
        # Loop through all ngrams in ngram_metrics
        for ngram, metrics in ngram_metrics.items():
            # Get the metrics for the current ngram
            label_metrics_dict = metrics[0]
            ngram_metrics_list = metrics[1]
            # Calculate the label percentage agreements for label_metrics_list
            label_percentage_agreements = self.calculate_label_percentage_agreements(label_metrics_dict)
            # Calculate the ngram percentage agreements for ngram_metrics_list
            ngram_percentage_agreements = self.calculate_ngram_percentage_agreements(ngram_metrics_list)
            # Calculate the label/ ngram percentage agreements for label_metrics_list
            label_ngram_percentage_agreements = self.calculate_label_ngram_percentage_agreements(label_percentage_agreements, ngram_percentage_agreements, ngram)
            # Add the metrics to the heat_map_metrics list
            heat_map_metrics.append(label_ngram_percentage_agreements)
        return heat_map_metrics

    def calculate_label_percentage_agreements(self, label_agreements: dict) -> dict:
        normalized_label_agreements = {}
        
        for label, kappa in label_agreements.items():
            normalized_kappa = (kappa + 1) / 2
            normalized_label_agreements[label] = normalized_kappa
        
        return normalized_label_agreements

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
            ngram_agreements[ngram] = ngram_agreements_list
        all_ngram_agreements = [[key] + list(agreement) for key, agreement in ngram_agreements.items()]
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

    def get_label_agreements_with_chance_cohen(self, dataframe: pd.DataFrame) -> dict:
        dataframe = dataframe.replace({None: "NoLabel"})
        annotators = [col for col in dataframe.columns if col.startswith('annotator')]

        unique_labels = pd.unique(dataframe.values.ravel('K'))

        label_agreement = {}

        for label in unique_labels:
            label_df = (dataframe == label).astype(int)

            total_kappa = 0
            total_pairs = 0

            for i, annotator1 in enumerate(annotators):
                for j, annotator2 in enumerate(annotators):
                    if i < j:
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            annotator1_labels = label_df[annotator1]
                            annotator2_labels = label_df[annotator2]
                            # Check if annotator1 and annotator2 labels are all 1
                            if all(annotator1_labels == 1) and all(annotator2_labels == 1):
                                kappa = 1.0
                            # Check if annotator1 and annotator2 labels are all 0
                            elif all(annotator1_labels == 0) and all(annotator2_labels == 0):
                                total_pairs += 1
                                continue
                            else:
                                kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)

                            # Check if there's a warning and if both annotators have variance in their labels
                            if w or (label_df[annotator1].nunique() <= 1 or label_df[annotator2].nunique() <= 1):
                                total_pairs += 1
                                continue

                        total_kappa += kappa
                        total_pairs += 1

            if total_pairs == 0:
                avg_kappa = float('nan')
            else:
                avg_kappa = total_kappa / total_pairs

            label_agreement[label] = avg_kappa

        return label_agreement 
  
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




                
                
                











    


