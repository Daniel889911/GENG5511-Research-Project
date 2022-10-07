from annotator import Annotator
import functions as fun

class Group_Metrics :

    def __init__(self, annotator1 : Annotator, annotator2:Annotator, annotator3:Annotator, annotator4:Annotator):

        self.annotator1 = annotator1
        self.annotator2 = annotator2
        self.annotator3 = annotator3
        self.annotator4 = annotator4
        self.annotator1_doc_ids = annotator1.get_doc_idxs()
        self.annotator2_doc_ids = annotator2.get_doc_idxs()
        self.annotator3_doc_ids = annotator3.get_doc_idxs()
        self.annotator4_doc_ids = annotator4.get_doc_idxs()
        self.same_docs1 = []
        self.same_docs2 = []
        self.same_docs = []
        self.same_mentions1 = []
        self.same_mentions2 = []
        self.token_list = []
        self.individual_mention_list = []
        self.individual_token_list = []
        self.group_mention_list = []
        self.group_token_list = []

    def get_same_doc_ids(self) :        
        for i in range(len(self.annotator1_doc_ids)):
            for j in range(len(self.annotator2_doc_ids)):
                if self.annotator1_doc_ids[i] == self.annotator2_doc_ids[j]:
                    self.same_docs1.append(self.annotator1_doc_ids[i])
                    break
        for i in range(len(self.annotator3_doc_ids)):
            for j in range(len(self.annotator4_doc_ids)):
                if self.annotator3_doc_ids[i] == self.annotator4_doc_ids[j]:
                    self.same_docs2.append(self.annotator3_doc_ids[i])
                    break
        for i in range(len(self.same_docs1)):
            for j in range(len(self.same_docs2)):
                if self.same_docs1[i] == self.same_docs2[j]:
                    self.same_docs.append(self.same_docs1[i])
                    break
        return self.same_docs

    def get_group_words(self):
        i = 40
        mention1 = self.annotator1.get_doc_mentions(i)
        token1 = self.annotator1.get_doc_tokens(i)
        annotated1 = fun.item_class(token1, mention1)  
        
        mention2 = self.annotator2.get_doc_mentions(i)
        token2 = self.annotator2.get_doc_tokens(i)
        annotated2 = fun.item_class(token2, mention2)

        mention3 = self.annotator3.get_doc_mentions(i)
        token3 = self.annotator3.get_doc_tokens(i)
        annotated3 = fun.item_class(token3, mention3)

        mention4 = self.annotator4.get_doc_mentions(i)
        token4 = self.annotator4.get_doc_tokens(i)
        annotated4 = fun.item_class(token4, mention4)

        return fun.majority_item(annotated1, annotated2, annotated3, annotated4)            
 


                
                
                











    


