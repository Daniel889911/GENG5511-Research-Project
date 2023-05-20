from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
from annotator import Annotator
from irrCAC.raw import CAC

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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label', 'ngram'])

        for annotator in self.annotator_list:
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label_ngrams(token, mention)
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'ngram'])
            temp_df['annotator_id'] = annotator_id
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)
        return annotated_df

    def create_single_annotations_table_ngrams(self, annotated_df):
        pivot_df = annotated_df.pivot_table(index=['token', 'ngram'], columns='annotator_id', values='label', aggfunc='first')
        pivot_df.reset_index(inplace=True)
        result_df = pivot_df.set_index('token')
        result_df = result_df.fillna('No Label')
        return result_df

    def get_accumulated_table_ngrams(self) -> pd.DataFrame:
        """
            Get the accumulated table for all the documents

            Returns:
                The accumulated table for all the documents
        """
        same_docs = self.get_same_doc_ids()
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            annotated_df = self.get_all_annotators_tokens_labels_single_doc_ngrams(doc_idx)
            table = self.create_single_annotations_table_ngrams(annotated_df)
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table
  
    def split_df_into_dfngrams(self, df):
        ngrams_dfs = {}    
        for ngram in df['ngram'].unique():
            ngrams_dfs[ngram] = df[df['ngram'] == ngram]
            ngrams_dfs[ngram] = ngrams_dfs[ngram].drop('ngram', axis=1)
        return ngrams_dfs
    
    def get_labels_and_ngrams(self, df):
        ngrams_dfs = {}
        ngram_metrics = {}
        ngrams_dfs = self.split_df_into_dfngrams(df)
        for ngram, ngram_df in ngrams_dfs.items():
            label_metrics_list = self.calculate_fleiss_kappas(ngram_df)
            ngram_metrics_list = self.get_single_ngram_agreement_list(ngram_df)
            ngram_metrics[ngram] = label_metrics_list, ngram_metrics_list
        return ngram_metrics    
 
    def calculate_ngram_percentage_agreements(self, ngram_metrics_list):
        ngram_percentage_metrics = ngram_metrics_list[0] / (ngram_metrics_list[0] + ngram_metrics_list[1])
        return ngram_percentage_metrics        
    
    def calculate_label_ngram_percentage_agreements(self, label_percentage_agreements, ngram_percentage_agreements, ngram):
        label_ngram_percentage_metrics = {}
        for label, metrics in label_percentage_agreements.items():
            combined_metrics = (metrics + ngram_percentage_agreements)/2
            label_ngram_percentage_metrics[label] = ngram, combined_metrics
        return label_ngram_percentage_metrics
    
    def convert_labels_ngrams_metrics_for_heat_map(self, ngram_metrics):
        heat_map_metrics = []
        for ngram, metrics in ngram_metrics.items():
            label_metrics_dict = metrics[0]
            ngram_metrics_list = metrics[1]
            label_percentage_agreements = self.calculate_label_percentage_agreements(label_metrics_dict)
            ngram_percentage_agreements = self.calculate_ngram_percentage_agreements(ngram_metrics_list)
            label_ngram_percentage_agreements = self.calculate_label_ngram_percentage_agreements(label_percentage_agreements, ngram_percentage_agreements, ngram)
            heat_map_metrics.append(label_ngram_percentage_agreements)
        return heat_map_metrics

    def calculate_label_percentage_agreements(self, label_agreements: dict) -> dict:
        normalized_label_agreements = {}        
        for label, kappa in label_agreements.items():
            normalized_kappa = (kappa + 1) / 2
            normalized_label_agreements[label] = normalized_kappa        
        return normalized_label_agreements

    def get_all_ngrams_agreements_lists(self, df):
        ngrams_dfs = {}
        ngram_agreements = {}
        ngrams_dfs = self.split_df_into_dfngrams(df)
        for ngram, ngram_df in ngrams_dfs.items():
            ngram_agreements_list = self.get_single_ngram_agreement_list(ngram_df)
            ngram_agreements[ngram] = ngram_agreements_list
        all_ngram_agreements = [[key] + list(agreement) for key, agreement in ngram_agreements.items()]
        return all_ngram_agreements 
    
    def get_single_ngram_agreement_list(self, df):         
        partial_agreements = 0
        full_agreements = 0        
        for row in range(len(df.index)):
            row_values = df.iloc[row].values
            none_count = (row_values == 'No Label').sum()            
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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label'])
        for annotator in self.annotator_list:
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label_labels(token, mention)
            temp_df = pd.DataFrame(annotated, columns=['token', 'label'])
            temp_df['annotator_id'] = annotator_id
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)
        return annotated_df

    def create_single_annotations_table_labels(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')
        table = table.astype(object).where(pd.notnull(table), None)
        return table

    def get_accumulated_table_labels(self) -> pd.DataFrame:
        """
            Get the accumulated table for all the documents

            Returns:
                The accumulated table for all the documents
        """
        same_docs = self.get_same_doc_ids()
        accumulated_table = pd.DataFrame()
        for doc_idx in same_docs:
            annotated_df = self.get_all_annotators_tokens_labels_single_doc_labels(doc_idx)
            table = self.create_single_annotations_table_labels(annotated_df)
            accumulated_table = pd.concat([accumulated_table, table], axis=0)
        return accumulated_table

    def calculate_fleiss_kappas(self, df):
        df = df.fillna('No Label')      
        unique_labels = set(df.values.ravel()) - {None}
        kappas = {}        
        for label in unique_labels:
            binary_annotations = df.applymap(lambda x: 1 if x == label else 0)            
            try:
                fleiss_kappa = CAC(binary_annotations)
                fleiss_kappa_values = fleiss_kappa.fleiss()
                kappa = fleiss_kappa_values['est']['coefficient_value']
            except ZeroDivisionError:
                kappa = np.nan
            kappas[label] = kappa        
        return kappas
  
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




                
                
                











    


