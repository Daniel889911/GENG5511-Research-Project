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
        annotated_df = pd.DataFrame(columns=['annotator_id', 'token', 'label', 'start', 'end'])
        for annotator in self.annotator_list:
            annotator_id = annotator.name
            mention = annotator.get_doc_mentions(doc_idx)
            token = annotator.get_doc_tokens(doc_idx)
            annotated = self.get_token_label(token, mention)
            temp_df = pd.DataFrame(annotated, columns=['token', 'label', 'start', 'end'])
            temp_df['annotator_id'] = annotator_id
            annotated_df = pd.concat([annotated_df, temp_df], ignore_index=True)
        return annotated_df

    def create_single_annotations_table(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        table = annotated_df.pivot_table(index='token', columns='annotator_id', values='label', aggfunc='first')
        table = table.astype(object).where(pd.notnull(table), None)
        return table

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
 
    def get_token_iaa_values(self, df: pd.DataFrame) -> pd.DataFrame:
        tokens = df.index.unique().values
        iaa_results = {}
        for token in tokens:
            token_data = df.loc[[token]]
            token_data.fillna("No Label", inplace=True)
            try:
                cac_coefficient = CAC(token_data)
                krippendorff_values = cac_coefficient.krippendorff()
                alpha = krippendorff_values['est']['coefficient_value']
            except ZeroDivisionError:
                alpha = np.nan
            iaa_results[token] = alpha
        return iaa_results

    def create_agreement_summary(self, agreements_dict):
        agreement_ranges = {
                    "low agreement": (-1.0, -0.6),
                    "medium low agreement": (-0.6, -0.2),
                    "medium agreement": (-0.2, 0.2),
                    "medium high agreement": (0.2, 0.6),
                    "high agreement": (0.6, 1.0)
                }

        summary_data = {key: [] for key in agreement_ranges.keys()}

        for label, kappa_score in agreements_dict.items():
            for range_name, (low, high) in agreement_ranges.items():
                if low <= kappa_score < high:
                    summary_data[range_name].append(label)

        summary_df = pd.DataFrame(dict([(k, pd.Series(v, dtype='object')) for k, v in summary_data.items()]))
        return summary_df

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

    




                
                
                











    


