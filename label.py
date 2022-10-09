from annotator import Annotator
from collections import Counter

class Label_Metrics :

    def __init__(self, annotator1 : Annotator, annotator2:Annotator, annotator3:Annotator, annotator4:Annotator):
        """
            Class for obtaining the annotator individual label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        self.annotator1 = annotator1
        self.annotator2 = annotator2
        self.annotator3 = annotator3
        self.annotator4 = annotator4
        self.annotator1_doc_ids = annotator1.get_doc_idxs()
        self.annotator2_doc_ids = annotator2.get_doc_idxs()
        self.annotator3_doc_ids = annotator3.get_doc_idxs()
        self.annotator4_doc_ids = annotator4.get_doc_idxs()
        self.same_docs = []

    def get_same_doc_ids(self) : 
        """
            Gets all the same annotated document ids for all the annotators

            Returns:
                The same annotated document ids annotated by all the annotators 
              
        """       
        list1 = self.get_common_files(self.annotator1_doc_ids,self.annotator2_doc_ids)
        list2 = self.get_common_files(self.annotator3_doc_ids,self.annotator4_doc_ids)
        self.same_docs = self.get_common_files(list1,list2)
        return self.same_docs
    
    def get_common_files(self, doc_ids1 : list, doc_ids2: list) -> list:
        """
            Gets all the same document ids for two annotators

            Parameters:
                doc_idx :
                    The document ids of two annotators
            Returns:
                The same document ids shared by two annotators 
              
        """
        doc_list = []
        for i in range(len(doc_ids1)):
            for j in range(len(doc_ids2)):
                if doc_ids1[i] == doc_ids2[j]:
                    doc_list.append(doc_ids1[i])  
                    break
        return doc_list 
    
    def get_token_label(self, tokens:list, mentions:list) -> list:
        """
            Gets the  

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        annotations_list1 = []
        annotations_list2 = []
        for ment in mentions:
            start = ment["start"]
            end = ment["end"]
            token = tokens[start:end]
            label = ment["labels"]
            token = self.list_To_String(token)
            annotations_list1 = [token, label, start, end]    
            annotations_list2.append(annotations_list1)    
        return annotations_list2
    
    def get_corpus_metrics(self):
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        annotated_doc = []
        annotated_corpus = []

        for i in self.same_docs:
            mention1 = self.annotator1.get_doc_mentions(i)
            token1 = self.annotator1.get_doc_tokens(i)
            annotated1 = self.get_token_label(token1, mention1)  
            annotated_doc.append(annotated1)

            mention2 = self.annotator2.get_doc_mentions(i)
            token2 = self.annotator2.get_doc_tokens(i)
            annotated2 = self.get_token_label(token2, mention2)
            annotated_doc.append(annotated2)

            mention3 = self.annotator3.get_doc_mentions(i)
            token3 = self.annotator3.get_doc_tokens(i)
            annotated3 = self.get_token_label(token3, mention3)
            annotated_doc.append(annotated3)

            mention4 = self.annotator4.get_doc_mentions(i)
            token4 = self.annotator4.get_doc_tokens(i)
            annotated4 = self.get_token_label(token4, mention4)
            annotated_doc.append(annotated4)

            doc_metrics = self.get_doc_metrics(annotated_doc)
            annotated_corpus.append(doc_metrics)
            annotated_doc.clear()
        return annotated_corpus

    def get_doc_metrics(self, group_annotated_doc: list) -> list :
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
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
            longest_label = self.list_To_String(longest_tokens_annotator[i][1])
            token_labels.append(longest_label)
            for annotator in group_annotated_doc:
                for annotated_token in annotator:
                     if longest_token in annotated_token[0]:
                        label = self.list_To_String(annotated_token[1]) 
                        token_labels.append(label)
                        break
            majority_label = self.get_majority_label(token_labels)
            all_labels_same = self.get_all_labels_same(token_labels)
            token_majority_label_metric = [longest_token, majority_label, all_labels_same]
            token_labels.clear()
            doc_metrics.append(token_majority_label_metric)
        return doc_metrics
    
    def get_annotator_metrics(self, annotated_corpus):
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        for annotated_document in annotated_corpus:
            print("new doc")
            for token_agreement in annotated_document:
                print(token_agreement[2])
 
    def get_majority_label(self, List: list) -> str:
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        counter = 0
        num = List[0]     
        for i in List:
            curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i 
        return num 

    def get_all_labels_same(self, List: list) -> bool:
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        for i in range(len(List)-1):
            if List[i] == List[i+1]:
                continue
            else:
                return False
        return True

    def list_To_String(self, List: list) -> str:
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """    
        str1 = " "    
        return (str1.join(List))

    def add_padding(self, length:int, List: list) -> list:
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        diff_len = length - len(List)
        if diff_len < 0:
            raise AttributeError('Length error list is too long')
        return List + ["null empty"] * diff_len       

    def get_longest_annotation_number(self, group_annotated_doc:list) -> int :
        """
            Class for obtaining the individual annotator label metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        longest = 0
        for i in range(len(group_annotated_doc)):
            length = len(group_annotated_doc[i])                 
            if length > longest:
                longest = length
                index = i
        return longest, index


                
                
                











    


