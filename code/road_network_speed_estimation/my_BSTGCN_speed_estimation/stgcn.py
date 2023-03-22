import os
import pickle

import numpy as np
import torch

from map_matching.utils.road_network import BaseRoadNetwork
from torch_geometric_temporal import StaticGraphTemporalSignal

from road_network_speed_estimation.utils import z_score


class STRoadGraph(StaticGraphTemporalSignal):
    def __init__(self, st_graph_data):
        graph_obj = BaseRoadNetwork("gcn")
        graph_obj.init_graph()
        graph_obj.road_graph.to_undirected()

        graph_attr = []  # [timestamp1, timestamp2,...]
        road_attr = []  # [(限速,车道数,长度),...]
        edge_attr = []  # [两个道路之间的行驶时间,...]

        for _timestamp, _graph_data in st_graph_data.items():
            graph_attr.append(_timestamp)
            free_speed_roads = [graph_obj.road_graph.nodes[_road_id]["free_speed"] for _road_id in graph_obj.road_graph.nodes]
            lanes_roads = [graph_obj.road_graph.nodes[_road_id]["lanes"] for _road_id in graph_obj.road_graph.nodes]
            length_roads = [graph_obj.road_graph.nodes[_road_id]["length"] for _road_id in graph_obj.road_graph.nodes]

            for one_road_attr in zip(free_speed_roads, lanes_roads, length_roads):
                road_attr.append(one_road_attr)

            for from_road_id, to_road_id in graph_obj.road_graph.edges:
                pass



        # 道路特征张量化
        node_features_transformed = torch.FloatTensor(z_score(node_features))
        labels_transformed = torch.FloatTensor(z_score(np.array(labels).reshape(-1, 1)))

        source_nodes = list(map(lambda x: int(x[0]), graph_obj.road_graph.edges))
        target_nodes = list(map(lambda x: int(x[1]), graph_obj.road_graph.edges))
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        for _timestamp, _graph_data in st_graph_data:
            pass
        super(STRoadGraph, self).__init__()


def get_st_road_graph(_index):
    with open(os.path.join(data_path, st_graph_files[_index]), "rb") as f:
        st_graph_data = pickle.load(f)
    return STRoadGraph(st_graph_data)


if __name__ == '__main__':
    data_path = "data"
    st_graph_files = os.listdir(data_path)
    #
    # with open(os.path.join(data_path, st_graph_files[0]), "rb") as f:
    #     file = pickle.load(f)
    #
    #     for timestamp, graph_data in file.items():
    #         print(timestamp, len(graph_data))

