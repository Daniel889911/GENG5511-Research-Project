from annotator import Annotator


class Calculator :

    def __init__(self, annotator1 : Annotator, annotator2:Annotator):

        self.annotator1_doc_ids = annotator1.get_doc_ids()
        self.annotator2_doc_ids = annotator2.get_doc_ids()
        self.same_docs = []
        self.same_mentions1 = []
        self.same_mentions2 = []


    def get_same_doc_ids(self) :        
        for i in range(len(self.annotator1_doc_ids)):
            for j in range(len(self.annotator2_doc_ids)):
                if self.annotator1_doc_ids[i] == self.annotator2_doc_ids[j]:
                    self.same_docs.append(self.annotator1_doc_ids[i])
                    break
        return self.same_docs

    def get_same_mentions(self):
        for i in self.same_docs:
            for j in self.annotator1_doc_ids:
                if i == j:
                    for k in range(len(self.annotator1.data)):
                        if self.annotator1.data[k]["doc_idx"] == i:
                            self.same_mentions1.append()
        for i in self.same_docs:
            for j in self.annotator2_doc_ids:
                if i == j:
                    self.same_mentions2.append(self.annotator2_doc_ids[i]["mentions"])

    


