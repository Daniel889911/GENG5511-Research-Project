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
 
    def create_rows_same_labels(self, start_row: int, end_row: int, df: pd.DataFrame) -> pd.DataFrame:
        """
            Creates a new DataFrame with the same label for all the annotators

            Parameters:
                rows :
                    The number of rows to be same in the DataFrame
                df :
                    The DataFrame with the labels for all the annotators

            Returns:
                The new DataFrame with the same label for all the annotators
        """
        new_df = df.iloc[start_row:end_row].copy()
        for row_idx in range(start_row, end_row):
            row = df.iloc[row_idx]
            mode = row.mode()
            if not mode.empty:
                most_common_label = mode[0]
            else:
                most_common_label = self.labels[0]
            for col_idx in range(df.shape[1]):
                new_df.iloc[row_idx, col_idx] = most_common_label
        return new_df

    def create_rows_different_labels(self, start_row: int, end_row: int, df: pd.DataFrame) -> pd.DataFrame:
        """
            Creates a new DataFrame with different labels for all the annotators

            Parameters:
                rows :
                    The number of rows to be different in the DataFrame
                df :
                    The DataFrame with the labels for all the annotators

            Returns:
                The new DataFrame with different labels for all the annotators

        """
        new_df = df.iloc[start_row:end_row].copy()
        for row_idx in range(new_df.shape[0]):
            used_labels = []
            used_labels = [new_df.iloc[row_idx, 0]]
            available_labels = [label for label in self.labels if label not in used_labels]

            for col_idx in range(1, df.shape[1]):
                current_label = df.iloc[row_idx, col_idx]

                if current_label in available_labels:
                    new_label = current_label
                else:
                    new_label = available_labels.pop(0)

                new_df.iloc[row_idx, col_idx] = new_label
                used_labels.append(new_label)
                available_labels = [label for label in self.labels if label not in used_labels]
        return new_df
    
    def calculate_same_different_row_numbers(self, percentage: float, df: pd.DataFrame) -> tuple:
        total_rows = len(df)
        same_rows = int(total_rows * percentage)
        different_rows = total_rows - same_rows
        return (same_rows, different_rows, total_rows) 
 
    def create_df_same_different(self, percentage: float, df: pd.DataFrame) -> pd.DataFrame:
        same_rows, different_rows, total_rows = self.calculate_same_different_row_numbers(percentage, df)
        same_label_df = self.create_rows_same_labels(0, same_rows, df)
        different_label_df = self.create_rows_different_labels(same_rows, total_rows, df)
        result_df = pd.concat([same_label_df, different_label_df], axis=0)
        return result_df

    def calculate_coefficient_for_table(self, accumulated_table) -> float:
        """
            Calculate Coefficients for all documents

            Returns:
                The Coefficient value for all documents
        """
        coefficients_dict = {}    

        # Initialise CAC with the accumulated_coefficients_table
        cac_coefficient = CAC(accumulated_table)

        # Calculate krippendorff coefficient value for the accumulated table
        krippendorff_values = cac_coefficient.krippendorff()
        krippendorff_alpha = krippendorff_values['est']['coefficient_value']

        fleiss_values = cac_coefficient.fleiss()
        fleiss_alpha =  fleiss_values['est']['coefficient_value']

        gwet_values = cac_coefficient.gwet()
        gwet_alpha =  gwet_values['est']['coefficient_value']

        conger_values = cac_coefficient.conger()
        conger_alpha =  conger_values['est']['coefficient_value']

        coefficients_dict['krippendorff'] = krippendorff_alpha
        coefficients_dict['fleiss'] = fleiss_alpha
        coefficients_dict['gwets'] = gwet_alpha
        coefficients_dict['cohens'] = conger_alpha

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

    def create_df_same_different2(self, percentage: float, df: pd.DataFrame) -> pd.DataFrame:
        same_rows, different_rows, total_rows = self.calculate_same_different_row_numbers(percentage, df)
        print(f'Creating a DataFrame with {same_rows} rows with same labels and {different_rows} rows with different labels')
        new_df = df.copy()        
        for row_idx in range(same_rows):
            row = new_df.iloc[row_idx]
            mode = row.mode()
            if not mode.empty:
                most_common_label = mode[0]
            else:
                most_common_label = row.iloc[0]
            for col_idx in range(new_df.shape[1]):
                new_df.iloc[row_idx, col_idx] = most_common_label
        for row_idx in range(same_rows, total_rows):
            used_labels = []
            used_labels = [new_df.iloc[row_idx, 0]]
            available_labels = [label for label in self.labels if label not in used_labels]

            for col_idx in range(1, df.shape[1]):
                current_label = df.iloc[row_idx, col_idx]

                if current_label in available_labels:
                    new_label = current_label
                else:
                    new_label = available_labels.pop(0)

                new_df.iloc[row_idx, col_idx] = new_label
                used_labels.append(new_label)
                available_labels = [label for label in self.labels if label not in used_labels]
        return new_df  
    




                
                
                











    


