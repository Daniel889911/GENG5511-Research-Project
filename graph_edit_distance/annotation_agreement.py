import pandas as pd
import numpy as np
import networkx as nx
import itertools

class AnnotationAgreement:
    @staticmethod
    def create_annotator_graph(data: pd.DataFrame, annotator: str) -> nx.Graph:
        G = nx.Graph()
        for index, row in data[data['annotator_id'] == annotator].iterrows():
            item = row['token']
            annotation = row['label']
            G.add_edge(item, annotation)
        return G

    @staticmethod
    def create_annotator_graphs(data: pd.DataFrame, annotator_nodes: set) -> dict:
        return {annotator: AnnotationAgreement.create_annotator_graph(data, annotator) for annotator in annotator_nodes}

    @staticmethod
    def calculate_pairwise_ged(annotator_graphs: dict, annotator_nodes: set) -> dict:
        pairwise_ged = {}
        for annotator1 in annotator_nodes:
            for annotator2 in annotator_nodes:
                if annotator1 == annotator2:
                    continue
                pair_key = tuple(sorted([annotator1, annotator2]))
                if pair_key in pairwise_ged:
                    continue
                ged = nx.graph_edit_distance(annotator_graphs[annotator1], annotator_graphs[annotator2])
                pairwise_ged[pair_key] = ged
        return pairwise_ged
