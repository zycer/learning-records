import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class TrafficDataset(StaticGraphTemporalSignal):
    def __init__(self, edge_index, edge_attr, features, targets):
        self.features = features
        self.targets = targets
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        super(TrafficDataset, self).__init__()

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        edge_index = self.edge_index
        edge_attr = self.edge_attr
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return self.features.shape[0]

# 随机生成8个节点8条边的图G
num_nodes = 8
num_edges = 8
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

# 随机生成边权重
edge_attr = torch.rand(num_edges)

# 随机生成节点特征，每个节点有5个特征，假设有10个时间步
num_features = 5
num_timesteps = 10
node_features = torch.rand(num_timesteps, num_nodes, num_features)

# 随机生成目标行驶时间，假设有10个时间步
targets = torch.rand(num_timesteps, num_nodes)

# 创建时空图数据集
dataset = TrafficDataset(edge_index, edge_attr, node_features, targets)

# 划分训练集、验证集和测试集
train_dataset, val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7, val_ratio=0.2)

# 创建 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
