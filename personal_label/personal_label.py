import pandas as pd
import numpy as np
from collections import Counter
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
    
    def get_annotators_labels_agreements(self, df):
        """
            Gets the annotator label agreements 
            Parameters:
                Dataframe :
                    The dataframe with the tokens, labels and annotators                    
            Returns:
                A list of dictionaries containing the annotator label agreements                
        """  
        df = df.fillna('NoLabel')
        all_annotator_agreements = []
        for annotator in df.columns:
            annotator_agreements = {}
            for token, token_data in df.iterrows():
                most_common_label = Counter(token_data.values).most_common(1)[0][0]
                if token_data[annotator] == most_common_label:
                    if most_common_label in annotator_agreements:
                        annotator_agreements[most_common_label][1] += 1
                    else:
                        annotator_agreements[most_common_label] = [most_common_label, 1, 0]
                else:
                    if most_common_label in annotator_agreements:
                        annotator_agreements[most_common_label][2] += 1
                    else:
                        annotator_agreements[most_common_label] = [most_common_label, 0, 1]
            all_annotator_agreements.append({annotator: list(annotator_agreements.values())})
        return all_annotator_agreements

    def get_agreement_percentages(self, data):
        new_data = []
        for annotator_data in data:
            annotator = list(annotator_data.keys())[0]
            labels_data = annotator_data[annotator]
            new_labels_data = []

            for label_data in labels_data:
                key = label_data[0]
                agreement = label_data[1]
                disagreement = label_data[2]
                percentage_agreement = agreement / (agreement + disagreement)
                new_labels_data.append([key, percentage_agreement])

            new_data.append({annotator: new_labels_data})

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
            labels_data = annotator_data[annotator]

            summary_data = {key: [] for key in agreement_ranges.keys()}

            for label_data in labels_data:
                label = label_data[0]
                agreement_percentage = label_data[1] * 100

                for range_name, (low, high) in agreement_ranges.items():
                    if low <= agreement_percentage <= high:
                        summary_data[range_name].append(label)

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




                
                
                











    


