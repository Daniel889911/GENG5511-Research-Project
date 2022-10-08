import json
import os

class Annotator:

    def __init__(self, name:str, person_annotations_file:str):
        """
            Parameters:
                name:
                    Name of the annotator
                person_annotations_file:
                    Location of person's annotated file                
        """
        self.name = name
        self.person_annotations_filepath = os.path.join('data', person_annotations_file)
        with open(self.person_annotations_filepath) as json_file:
            self.annotations = [json.loads(line) for line in open(self.person_annotations_filepath,'r')]
        self.doc_idxs = []

    def get_doc_idxs(self) -> list :        
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            self.doc_idxs.append(ids)
        return self.doc_idxs
    
    def get_doc_idx(self, doc_idx: int) -> dict :
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]

    def get_doc_mentions(self, doc_idx: int) -> dict :
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]["mentions"]

    def get_doc_tokens(self, doc_idx: int) -> list :
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]["tokens"]
   
            
