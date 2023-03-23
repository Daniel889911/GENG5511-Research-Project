from annotator import Annotator
from statistics import multimode

class NgramIndividualMetrics :

    def __init__(self, *args):
        """
            Class for obtaining the individual ngram metrics 

            Parameters:
                annotators :
                    Instance of Annotator class for a person
              
        """
        self.annotator_list = list(args)
        self.annotator_count = len(self.annotator_list)
        for i in range(len(self.annotator_list)):
            setattr(self, f'annotator{i + 1}', self.annotator_list[i])
        for i in range(len(self.annotator_list)):
            setattr(self, f'annotator{i + 1}_doc_ids', self.annotator_list[i].get_doc_idxs())
        self.same_docs = []

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

    
    def get_token_label(self, tokens:list, mentions:list) -> list:
        """
            Gets all the separated tokens with labels into a list 

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
            for annotator in self.annotator_list:
                mention = annotator.get_doc_mentions(i)
                token = annotator.get_doc_tokens(i)
                annotated = self.get_token_label(token, mention)
                annotated_doc.append(annotated)
            single_doc_metrics = self.get_single_doc_metrics(annotated_doc)
            annotated_corpus.append(single_doc_metrics)
            annotated_doc.clear()
        return annotated_corpus

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
    
    def get_individual_annotations(self) -> list:
        annotated_doc= []
        for i in self.same_docs:    
            mention = self.annotator1.get_doc_mentions(i)
            token = self.annotator1.get_doc_tokens(i)
            annotated = self.get_token_label(token, mention)  
            annotated_doc.append(annotated)
        return annotated_doc
    
    def get_individual_ngram_metrics(self) -> list:

        individual_list = self.get_individual_annotations()
        group_list = self.get_corpus_metrics()
        number_group_docs = self.get_number_group_docs()

        ind_count = 0

        for count1, ind_documents in enumerate(individual_list):
            for count2, group_documents in enumerate(group_list) :
                if count1 == count2 :
                    for count3, ind_metrics in enumerate(ind_documents):
                        for count4, group_metrics in enumerate(group_documents):
                            if count3 == count4:
                                if ind_metrics[2] == group_metrics[0] and group_metrics[2] >= 0.5:
                                    ind_count += 1       
        ind_stats = [self.annotator1.name, ind_count, number_group_docs-ind_count]
        return ind_stats

    def get_number_group_docs(self) -> int:       

        group_list = self.get_corpus_metrics()

        number = 0
        new_number = 0

        for document in group_list:
            number = len(document)
            new_number += number
        return new_number

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

    def get_all_ngrams(self) -> list:
        """
            Gets all the labels in the annotated corpus 
                  
            Returns:
                All the labels in the annotated corpus in a list 
              
        """
        ngram_list = []

        annotated_corpus = self.get_corpus_metrics()

        for document_metrics in annotated_corpus :
            for metrics in document_metrics:
                if metrics[0][2] not in ngram_list:
                    ngram_list.append(metrics[0][2])
        return ngram_list

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

     
                
