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
            self.data = [json.loads(line) for line in open(self.person_annotations_filepath,'r')]

    def get_doc_ids(self) -> list :
        doc_ids = []
        for idx in range(len(self.data)):
            ids = self.data[idx]["doc_idx"]
            doc_ids.append(ids)
        return doc_ids

    def get_mentions(self, doc_id: int) -> str :
        return self.data[doc_id]["mentions"]
   
            
