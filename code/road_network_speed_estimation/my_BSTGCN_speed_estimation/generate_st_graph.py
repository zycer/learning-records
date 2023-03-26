import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from map_matching.utils.road_network import BaseRoadNetwork


y_standard_scaler = StandardScaler()
edge_standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()


class STRoadGraph:
    def __init__(self, st_graph_data):
        graph_obj = BaseRoadNetwork("gcn")
        graph_obj.init_graph()
        graph_obj.road_graph.to_undirected()

        graph_attrs = []
        road_attrs = []
        edge_attrs = []

        for _timestamp, _graph_data in st_graph_data.items():
            graph_attrs.append(_timestamp)  # [timestamp1, timestamp2,...]
            road_attr = []  # [(限速,车道数,长度),...]
            edge_attr = []  # [两个道路之间的行驶时间,...]

            road_free_speeds = []
            road_lanes = []
            road_lengths = []
            road_types = []
            road_travel_times = []

            for _road_id in graph_obj.road_graph.nodes:
                road_free_speeds.append(graph_obj.road_graph.nodes[_road_id]["free_speed"])
                road_lanes.append(graph_obj.road_graph.nodes[_road_id]["lanes"])
                # road_lengths.append(graph_obj.road_graph.nodes[_road_id]["length"])
                road_types.append(graph_obj.road_graph.nodes[_road_id]["type_name"])
                # road_travel_times.append(graph_obj.road_graph.nodes[_road_id]["length"] / _graph_data[_road_id])

            road_types_one_hot = pd.get_dummies(road_types).values  # 独热编码

            for one_road_attr in zip(road_free_speeds, road_lanes):
                road_attr.append(one_road_attr)

            # 将道路类型的独热编码与其他特征连接起来
            road_attr = np.concatenate((road_attr, road_types_one_hot), axis=1)

            for from_road_id, to_road_id in graph_obj.road_graph.edges:
                road_travel_times.append(graph_obj.road_graph.nodes[from_road_id]["length"] / _graph_data[from_road_id])
                road_lengths.append(graph_obj.road_graph.nodes[from_road_id]["length"])

            for one_edge_attr in zip(road_lengths, road_travel_times):
                edge_attr.append(one_edge_attr)

            road_attrs.append(road_attr)
            edge_attrs.append(edge_attr)

        source_nodes = list(map(lambda x: int(x[0]), graph_obj.road_graph.edges))
        target_nodes = list(map(lambda x: int(x[1]), graph_obj.road_graph.edges))

        # 特征transformed并张量化
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        self.edge_attrs = torch.DoubleTensor(edge_attrs)
        self.road_attrs = torch.DoubleTensor(np.array(road_attrs))
        self.graph_attrs = torch.tensor(graph_attrs, dtype=torch.long)

    def __getitem__(self, index):
        x = self.road_attrs[index]
        edge_index = self.edge_index
        edge_attr = self.edge_attrs[index]
        return Data(x=torch.tensor(min_max_scaler.fit_transform(x), dtype=torch.double),
                    edge_index=edge_index, graph_attr=self.graph_attrs[index],
                    edge_attr=torch.tensor(
                        edge_standard_scaler.fit_transform(edge_attr), dtype=torch.double))

    def __len__(self):
        return len(self.graph_attrs)


class STGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(STGraphDataset, self).__init__()
        self.data_list = data_list

    def _process(self):
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_st_road_graph(data_path):
    with open(data_path, "rb") as f:
        st_graph_data = pickle.load(f)
    return STRoadGraph(st_graph_data)


def get_st_graph_loader(data_path, batch_size=3):
    with open(data_path, "rb") as f:
        st_graph_data = pickle.load(f)
        st_graph_dataset = STGraphDataset(STRoadGraph(st_graph_data))
    return DataLoader(st_graph_dataset, batch_size=batch_size, shuffle=True)

