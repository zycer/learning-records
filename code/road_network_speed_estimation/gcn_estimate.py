import os.path as osp
import os

import networkx as nx
import torch.nn

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler


def z_score(raw_data):
    """
    对原始特征进行z_score标准化
    """
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(raw_data)


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
            for node in road_network_graph.nodes:
                node_attr = road_network_graph.nodes[node]
                del node_attr["from_node_id"]
                del node_attr["to_node_id"]
                node_attr = list(node_attr.values())
                node_features.append(node_attr)

            # 道路特征张量化
            node_features_transformed = torch.FloatTensor(z_score(node_features))

            source_nodes = list(map(lambda x: int(x[0]), road_network_graph.edges))
            target_nodes = list(map(lambda x: int(x[1]), road_network_graph.edges))
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            graph_data = Data(x=node_features_transformed, edge_index=edge_index)

            data_list.append(graph_data)
            print(f"Processed {osp.basename(one_graph_path)}")

        # 持久化处理后的数据
        graph_data, slices = self.collate(data_list)
        torch.save((graph_data, slices), self.processed_paths[0])

    #
    # class GCN(torch.nn.Module):
    #     def __init__(self, hidden_channels):
    #         super().__init__()
    #         torch.manual_seed(112233)
    #         self.conv1 = GCNConv(datasets.num_features, hidden_channels)
    #         self.conv2 = GCNConv(hidden_channels, datasets.num_classes)
    #
    #     def forward(self, x, edge_index):
    #         x = self.conv1(x, edge_index)
    #         x = x.relu()
    #         x = F.dropout(x, p=0.5, trainning=self.training)
    #         x = self.conv2(x, edge_index)
    #
    #         return x
    #
    #
    # def train():
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data.x, data.edge_index)
    #     loss = criterion(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    #
    #     return loss
    #
    #
    # def test():
    #     model.eval()
    #     out = model(data.x, data.edge_index)
    #     pred = out.argmax(dim=1)
    #     test_correct = pred[data.test_mask] == data.y[data.train_mask]
    #     test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    #
    #     return test_acc


if __name__ == '__main__':
    # model = GCN(hidden_channels=16)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    # data = []
    road_graph_data = RoadNetworkGraphData()
    data_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)

    for data in data_loader:
        print(data, type(data))

