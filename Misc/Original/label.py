from annotator import Annotator
from statistics import multimode

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
    
    def get_corpus_metrics(self):
        """
            Gets the calculated label corpus metrics from all the documents from the group

            Returns:
                The calculated label corpus metrics for all documents annotated by the group
              
        """
        annotated_doc = []

        self.get_same_doc_ids()

        for i in self.same_docs:  
            for annotator in self.annotator_list:
                mention = annotator.get_doc_mentions(i)
                token = annotator.get_doc_tokens(i)
                annotated = self.get_token_label(token, mention)
                annotated_doc.append(annotated)
            single_doc_metrics = self.get_single_doc_metrics(annotated_doc)
            self.annotated_corpus.append(single_doc_metrics)
            annotated_doc.clear()

    def get_single_doc_metrics(self, group_annotated_doc: list) -> list :
        """
            Gets the calculated label corpus metrics from all the same documents from the group

            Parameters:
                group_annotated_doc :
                    The annotated group documents containing tokens with labels for the same document
                    
            Returns:
                The calculated corpus label metrics for a single document
              
        """
        token_labels = []
        doc_metrics = []
        longest_tokens, number = self.get_longest_annotation_number(group_annotated_doc)
        for i in range(len(group_annotated_doc)):
            if i != number:
                group_annotated_doc[i] = self.add_padding(longest_tokens, group_annotated_doc[i])
        longest_tokens_annotator = group_annotated_doc[number]
        group_annotated_doc.pop(number)
        for i in range(len(longest_tokens_annotator)):            
            longest_token = longest_tokens_annotator[i][0]
            longest_label = longest_tokens_annotator[i][1]
            token_labels.append(longest_label)
            for annotator in group_annotated_doc:
                for annotated_token in annotator:
                    if longest_token in annotated_token[0]:
                        label = annotated_token[1]
                        token_labels.append(label)
                        break            
            majority_label = self.list_To_String(multimode(token_labels))            
            all_labels_same = self.get_all_labels_same(token_labels)
            token_majority_label_metric = [longest_token, majority_label, all_labels_same]
            token_labels.clear()
            doc_metrics.append(token_majority_label_metric)
        return doc_metrics
    
    def get_label_metrics(self) -> list:
        """
            Gets the label category metrics for corpus 

            Parameters:
                annotated_corpus :
                    The annotated corpus containing processed label metrics
          
            Returns : 
                The calculated metrics for all the label             
        """
        label_list = self.get_all_labels()
        label_metrics_list = []
        counter_false = 0
        counter_true = 0
        for label in label_list :
            for annotated_document in self.annotated_corpus:
                for token_agreement in annotated_document:
                    if token_agreement[1] == label:
                        if token_agreement[2] == True:
                            counter_true += 1
                        if token_agreement[2] == False:
                            counter_false += 1
            label_metrics = [label, counter_true, counter_false]
            label_metrics_list.append(label_metrics)
        return label_metrics_list

    def get_all_labels(self) -> list:
        """
            Gets all the labels in the annotated corpus 
                  
            Returns:
                All the labels in the annotated corpus in a list 
              
        """
        label_list = []
        for document_metrics in self.annotated_corpus :
            for metrics in document_metrics:
                if metrics[1] not in label_list:
                    label_list.append(metrics[1])
        return label_list
 
    def get_all_labels_same(self, Label_List: list) -> bool:
        """
            Gives a boolean True if all labels are the same and False if not all the same for a token 

            Parameters:
                Label_List :
                    The list of labels annotated for a single token by the group
                    
            Returns:
                True if all labels are the same and False if not all the same 
              
        """
        for i in range(len(Label_List)-1):
            if Label_List[i] == Label_List[i+1]:
                continue
            else:
                return False
        return True

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

    def add_padding(self, list_length:int, token_list: list) -> list:
        """
            Adds padding to shorter list of tokens in order to be able to loop through all lists together

            Parameters:
                list_length :
                    The desired length of the list to add padding to
                token_list : 
                    The token list to add padding into the desired list length
                    
            Returns:
                The token list with padding into the desired list length 
              
        """
        diff_len = list_length - len(token_list)
        if diff_len < 0:
            raise AttributeError('Length error list is too long')
        return token_list + ["null empty"] * diff_len       

    def get_longest_annotation_number(self, group_annotated_token_doc:list) -> int :
        """
            Gets the longest list length of the token in the group annotated document 

            Parameters:
                group_annotated_token_doc :
                    The group annotation tokens for a document 
                    
            Returns:
                The longest list length of token in the document
              
        """
        longest = 0
        for i in range(len(group_annotated_token_doc)):
            length = len(group_annotated_token_doc[i])                 
            if length > longest:
                longest = length
                index = i
        return longest, index


                
                
                











    


