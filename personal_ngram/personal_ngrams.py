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
        same_docs = set(self.annotator_list[0].get_doc_idxs())

        for annotator in self.annotator_list[1:]:
            same_docs.intersection_update(annotator.get_doc_idxs())
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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label', 'ngram'])
        for annotator in self.annotator_list:
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'ngram'])
            temp_df['annotator_id'] = annotator_id
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)
        return annotated_df
    
    def create_single_annotations_table(self, annotated_df):
        pivot_df = annotated_df.pivot_table(index=['token', 'ngram'], columns='annotator_id', values='label', aggfunc='first')
        pivot_df.reset_index(inplace=True)
        result_df = pivot_df.set_index('token')
        result_df.fillna('None', inplace=True)
        return result_df

    def get_accumulated_table(self) -> pd.DataFrame:
        """
            Get the accumulated table for all the documents

            Returns:
                The accumulated table for all the documents
        """
        same_docs = self.get_same_doc_ids()
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            annotated_df = self.get_all_annotators_tokens_labels_single_doc(doc_idx)
            table = self.create_single_annotations_table(annotated_df)
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table

    def get_annotator_ngrams_agreements_lists(self, df):
        annotator_ngrams_list = {col: [] for col in df.columns if col != 'ngram'}

        for ngram in df['ngram'].unique():
            ngram_df = df[df['ngram'] == ngram].drop('ngram', axis=1)
            annotator_ngrams = {col: {'agree': 0, 'disagree': 0} for col in ngram_df.columns}
            for row_idx in range(len(ngram_df)):
                row_values = ngram_df.iloc[row_idx]
                majority_agreement = (row_values != 'None').sum() >= len(row_values) / 2
                for col_idx in range(len(ngram_df.columns)):
                    col = ngram_df.columns[col_idx]
                    if ngram_df.iloc[row_idx, col_idx] != 'None':
                        if majority_agreement:
                            annotator_ngrams[col]['agree'] += 1
                        else:
                            annotator_ngrams[col]['disagree'] += 1
                    else:
                        if majority_agreement:
                            annotator_ngrams[col]['disagree'] += 1
                        else:
                            annotator_ngrams[col]['agree'] += 1
            for annotator, ngram_list in annotator_ngrams.items():
                annotator_ngrams_list[annotator].append([
                    f'{ngram}-ngram',
                    ngram_list['agree'],
                    ngram_list['disagree']
                ])

        return annotator_ngrams_list               
        

    def get_agreement_percentages(self, data):
        new_data = []
        for annotator, labels_data in data.items():
            new_labels_data = []
            for label_data in labels_data:
                key = label_data[0]
                agreement = label_data[1]
                disagreement = label_data[2]
                if agreement + disagreement > 0:
                    percentage_agreement = agreement / (agreement + disagreement)
                else:
                    percentage_agreement = 0.0
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




                
                
                











    


