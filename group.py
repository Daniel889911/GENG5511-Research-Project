from annotator import Annotator


class Group_Metrics :

    def __init__(self, annotator1 : Annotator, annotator2:Annotator, annotator3:Annotator, annotator4:Annotator):

        self.annotator1_doc_ids = annotator1.get_doc_idxs()
        self.annotator2_doc_ids = annotator2.get_doc_idxs()
        self.annotator3_doc_ids = annotator3.get_doc_idxs()
        self.annotator4_doc_ids = annotator4.get_doc_idxs()
        self.same_docs1 = []
        self.same_docs2 = []
        self.same_docs = []
        self.same_mentions1 = []
        self.same_mentions2 = []


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

    


