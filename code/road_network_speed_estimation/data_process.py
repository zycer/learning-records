import os.path as osp
import os

import networkx as nx
import torch.nn
import numpy as np

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def z_score(raw_data):
    """
    对原始特征进行z_score标准化
    """
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(raw_data)


def visualize_graph(graph, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()


class RoadNetworkGraphData(InMemoryDataset):
    """
    自定义路网图结构
    """

    def __init__(self, root="data/run_data", transform=None, pre_transform=None):
        self.root = root
        super(RoadNetworkGraphData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        files = [file_path for file_path in os.listdir(self.raw_dir)]
        return files

    @property
    def processed_file_names(self):
        return ["multi_road_network_graph.pt"]

    def download(self):
        pass

    def process(self):
        """
        对原始路网图数据进行特征编码、标准化后转换为torch格式数据
        """
        data_list = []
        for one_graph_path in self.raw_paths:
            # 有向图转无向图
            road_network_graph = nx.read_graphml(one_graph_path).to_undirected()
            node_features = []
            labels = []
            for node in road_network_graph.nodes:
                node_attr = road_network_graph.nodes[node]
                labels.append(node_attr.pop("average_speed"))
                del node_attr["from_node_id"]
                del node_attr["to_node_id"]
                node_attr = list(node_attr.values())
                node_features.append(node_attr)

            # 道路特征张量化
            node_features_transformed = torch.FloatTensor(z_score(node_features))
            labels_transformed = torch.FloatTensor(z_score(np.array(labels).reshape(-1, 1)))

            source_nodes = list(map(lambda x: int(x[0]), road_network_graph.edges))
            target_nodes = list(map(lambda x: int(x[1]), road_network_graph.edges))
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            graph_data = Data(x=node_features_transformed, edge_index=edge_index, y=labels_transformed)

            data_list.append(graph_data)
            print(f"Processed {osp.basename(one_graph_path)}")

        # 持久化处理后的数据
        graph_data, slices = self.collate(data_list)
        torch.save((graph_data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # model = GCN(hidden_channels=16)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    # data = []
    road_graph_data = RoadNetworkGraphData()
    data_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)
    data = next(iter(data_loader))
    print(data)
    # visualize_graph(to_networkx(data, to_undirected=True), color=data.y)
