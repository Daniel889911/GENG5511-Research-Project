from annotator import Annotator
from statistics import multimode

class Ngram_Metrics :

    def __init__(self, annotator1 : Annotator, annotator2:Annotator, annotator3:Annotator, annotator4:Annotator):
        """
            Class for obtaining the individual ngram metrics 

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
        self.annotator_count = 4

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
                The list of same document ids shared by two annotators 
              
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
                tokens :
                    The list of tokens in a document id
                mentions : 
                    The list of mentions in a document id
            Returns:
                The same document ids shared by two annotators 
              
        """
        annotations_list1 = []
        annotations_list2 = []
        for ment in mentions:
            start = ment["start"]
            end = ment["end"]
            ngram = end - start
            token = tokens[start:end]
            label = ment["labels"]
            token = self.list_To_String(token)
            annotations_list1 = [token, label, [start, end, ngram]]    
            annotations_list2.append(annotations_list1)    
        return annotations_list2
    
    def get_corpus_metrics(self):
        """
            Gets the calculated label corpus metrics from all the documents from the group

            Returns:
                The calculated label corpus metrics for all documents annotated by the group
              
        """
        annotated_doc = []
        annotated_corpus = []

        self.get_same_doc_ids()

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
            Gets the calculated label corpus metrics from all the same documents from the group

            Parameters:
                group_annotated_doc :
                    The annotated group documents containing tokens with labels for the same document
                    
            Returns:
                The calculated corpus label metrics for a single document
              
        """
        token_labels = []
        doc_metrics = []
        majority_label = ''
        majority_label_count = 0

        for count1, orig_annotator in enumerate(group_annotated_doc):
            for orig_annotated_ngram in orig_annotator: 
                token_labels.clear()                       
                count_ngram = 1
                orig_token = orig_annotated_ngram[0]            
                orig_label = orig_annotated_ngram[1]
                if type(orig_label) == list:
                    orig_label = self.list_To_String(orig_label)
                token_labels.append(orig_label)
                orig_ngram = orig_annotated_ngram[2]            
                for count2, other_annotator in enumerate(group_annotated_doc):
                    if count1 != count2:
                        for other_annotated_ngram in other_annotator:
                            if orig_ngram == other_annotated_ngram[2]:                                                     
                                count_ngram += 1         
                                other_label = self.list_To_String(other_annotated_ngram[1])
                                token_labels.append(other_label)    
                                break
                if count_ngram < 2 and not (self.search_ngram_in_list(orig_ngram, doc_metrics)):
                    orig_ngram_ratio = count_ngram/ self.annotator_count
                    label_ratio = "N/A"
                    single_ngram = [orig_ngram, count_ngram, orig_ngram_ratio, orig_token, orig_label, label_ratio]
                    doc_metrics.append(single_ngram)
                else :
                    if not self.search_ngram_in_list(orig_ngram, doc_metrics):
                        majority_label = multimode(token_labels)
                        if type(majority_label) == list:
                            majority_label = self.list_To_String(majority_label)
                        majority_label_count = self.get_majority_label_count(token_labels, majority_label)
                        # Calculate the percentage ratio of ngrams 
                        ngram_ratio = count_ngram/ self.annotator_count                                     
                        # Calculate the percentage ratio of majority label
                        label_ratio = (majority_label_count)/ len(token_labels)
                        ngram_metrics = [orig_ngram, count_ngram, ngram_ratio, orig_token, majority_label, label_ratio]
                        doc_metrics.append(ngram_metrics)
                    else:
                        found_ngram, index = self.search_ngram(orig_ngram, doc_metrics)
                        if self.compare_ngram_metric(count_ngram, found_ngram):
                            doc_metrics.pop(index)
                            majority_label = multimode(token_labels)
                            if type(majority_label) == list:
                                majority_label = self.list_To_String(majority_label)
                            majority_label_count = self.get_majority_label_count(token_labels, majority_label)
                            # Calculate the percentage ratio of ngrams 
                            ngram_ratio = count_ngram/ self.annotator_count                                     
                            # Calculate the percentage ratio of majority label
                            label_ratio = (majority_label_count)/ len(token_labels)
                            ngram_metrics = [orig_ngram, count_ngram, ngram_ratio, orig_token, majority_label, label_ratio]
                            doc_metrics.append(ngram_metrics)
                        else:
                            continue                          
        return doc_metrics
    
    def get_label_ngram_metrics(self, corpus_metrics: list) -> list:
        """
            Get the label and ngram metrics

            Parameters:
                corpus_metrics :
                    The corpus metrics
            
            Returns :
                The list of calculated values of label and ngram metrics for graphing
              
        """
        labels_list = self.get_all_labels()

        metrics_list = []

        for label in labels_list:
            for document_metric in corpus_metrics:
                for ngram_metric in document_metric:
                    if ngram_metric[4] == label:
                        ngram_value = 0
                        label_value = 0
                        count_label = 0
                        count_ngram = 0
                        ngram = ngram_metric[0][2]
                        ngram_value += ngram_metric[2]
                        count_ngram += 1
                        if ngram_metric[5] == "N/A":
                            continue
                        else :
                            label_value += int(ngram_metric[5])
                            count_label += 1
            if count_label == 0 :
                label_average_value = "N/A"
            else :                
                label_average_value = label_value/ count_label
            ngram_average_value = ngram_value/ count_ngram
            if label_average_value == "N/A":
                continue
            else :
                overall_average_value = (ngram_average_value + label_average_value)/2
            label_ngram_metric = [label, ngram, overall_average_value]
            metrics_list.append(label_ngram_metric)

        return metrics_list
    
    def get_all_labels(self) -> list:
        """
            Gets all the labels in the annotated corpus 
                  
            Returns:
                All the labels in the annotated corpus in a list 
              
        """
        label_list = []

        annotated_corpus = self.get_corpus_metrics()

        for document_metrics in annotated_corpus :
            for metrics in document_metrics:
                if metrics[4] not in label_list:
                    label_list.append(metrics[4])

        return label_list

    def search_ngram(self, ngram: list, doc_metrics: list):
        """
            Searches an ngram metric in the document metric

            Parameters:
                ngram :
                    The ngram metric list
                    
            Returns:
                The found ngram metric and the index in doc_metrics
              
        """   
        for count, metric in enumerate(doc_metrics):
            if ngram in metric:
                return doc_metrics[count], count
    
    def compare_ngram_metric(self, count_ngram, ngram2):
        """
            Compares the count of ngrams for a ngram metric list

            Parameters:
                count_ngram :
                    The ngram count to be compared with
                ngram2 :
                    The ngram metric list
                    
            Returns:
                True if the ngram count is larger than the ngram count in the ngram metric list otherwise False
              
        """ 
        return (count_ngram > ngram2[1])

    def search_ngram_in_list(self, orig_ngram:list, doc_metrics: list) -> bool :
        """
            Searches an ngram metric in the document metric list

            Parameters:
                orig_ngram :
                    The ngram metric list to find
                doc_metrics :
                    The document metrics list to search for the ngram metric
                    
            Returns:
                True if the ngram metric is located in the document metric and False otherwise
              
        """ 
        for ngrams in doc_metrics:
            if orig_ngram in ngrams:
                return True
        return False
 
    def get_majority_label_count(self, List:list, majority_label: str) -> int:
        """
            Gets the count of the number of majority labels for the majority annotated label 

            Parameters:
                List :
                    The list of the majority labels
                majority_label :
                    The majority label to count in the List
                    
            Returns:
                The count of the number of majority labels in the list
              
        """ 
        counter = 0
        for label in List:
            if label in majority_label:
                counter += 1
        return counter
 
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

     
                
