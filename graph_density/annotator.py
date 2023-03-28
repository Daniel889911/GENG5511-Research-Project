import json
import os

class Annotator:

    def __init__(self, name:str, person_annotations_file:str):
        """
            Class for obtaining annotators annotations including doc ids, tokens, and mentions

            Parameters:
                name:
                    Name of the annotator

                person_annotations_file:
                    Location of annotator's annotated file                
        """
        self.name = name
        # Get the absolute path of the parent directory of the current script
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        # Set the path to the person annotations file, located one level up of the current script
        self.person_annotations_filepath = os.path.join(parent_dir, 'data', person_annotations_file)
        with open(self.person_annotations_filepath) as json_file:
            self.annotations = [json.loads(line) for line in open(self.person_annotations_filepath,'r')]
        self.doc_idxs = []

    def get_name(self) -> str :
        """
            Gets the name of the annotator 

            Returns:
                annotators name              
        """
        return self.name

    def get_doc_idxs(self) -> list :   
        """
            Gets all the document ids annotated by the annotator 

            Returns:
                A list of document ids annotated by the annotator  
              
        """     
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            self.doc_idxs.append(ids)
        return self.doc_idxs
    
    def get_doc_idx(self, doc_idx: int) -> dict :
        """
            Gets a particular document id with annotations from annotator

            Parameters:
                doc_idx :
                    The document id to get the annotations

            Returns:
                The document id dictionary of annotations from the annotator  
              
        """
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]

    def get_doc_mentions(self, doc_idx: int) -> dict :
        """
            Gets the document id with all annotated mentions from annotator 

            Parameters:
                doc_idx :
                    The document id to get the annotated mentions

            Returns:
                The doc id dictionary of all annotated mentions from the annotator  
              
        """
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]["mentions"]
            
    def get_doc_tokens(self, doc_idx: int) -> list :
        """
            Gets the document id with all annotated tokens from annotator

            Parameters:
                doc_idx :
                    The document id to get the annotated tokens

            Returns:
                The doc id list of all annotated tokens from the annotator  
              
        """
        for idx in range(len(self.annotations)):
            ids = self.annotations[idx]["doc_idx"]
            if ids == doc_idx:
                return self.annotations[idx]["tokens"]
           
