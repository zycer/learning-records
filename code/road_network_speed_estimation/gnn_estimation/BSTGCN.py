import numpy as np
import torch
from torch_geometric.data import Data

# 假设原始数据以列表形式存储，每个元素是一个包含道路特征和连接关系的字典
raw_data = [...]

# 获取节点特征
node_features = []
for item in raw_data:
    node_features.append([item['speed'], item['timestamp'], item['length'], item['lanes'], item['speed_limit']])
node_features = np.array(node_features)

# 获取边关系
edges = []
for item in raw_data:
    road_id = item['road_id']
    for connected_road in item['connected_roads']:
        edges.append((road_id, connected_road))
edges = np.array(edges)

# 创建时空特征矩阵
num_nodes, num_features = node_features.shape
window_size = 3  # 滑动窗口大小
stride = 1  # 滑动窗口步长

temporal_features = []
for i in range(0, num_nodes - window_size + 1, stride):
    temporal_features.append(node_features[i:i + window_size].T)
temporal_features = np.stack(temporal_features)

# 创建 PyG Data 对象
x = torch.tensor(temporal_features, dtype=torch.float)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
graph = Data(x=x, edge_index=edge_index)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

# 使用您提供的数据构建时空图
data_x = torch.tensor([...], dtype=torch.float)
data_edge_index = torch.tensor([...], dtype=torch.long)
graph = Data(x=data_x, edge_index=data_edge_index.t().contiguous())

# 定义STGCN层，结合BayesianGCNConv
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, Kt):
        super(STGCNLayer, self).__init__()
        self.Kt = Kt
        self.conv = BayesianGCNConv(in_channels, out_channels)
        self.t_conv = nn.Conv2d(in_channels, out_channels, (1, Kt))

    def forward(self, x, edge_index, edge_weight=None):
        spatial_gcn = self.conv(x, edge_index, edge_weight)
        temporal_gcn = self.t_conv(x.view(x.size(0), x.size(1), -1, self.Kt))
        return spatial_gcn + temporal_gcn.view_as(spatial_gcn)

# 定义STGCN模型
class STGCNModel(nn.Module):
    def __init__(self, num_features, hidden_size, Kt, output_size):
        super(STGCNModel, self).__init__()

        self.stgcn1 = STGCNLayer(num_features, hidden_size, Kt)
        self.stgcn2 = STGCNLayer(hidden_size, hidden_size, Kt)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.stgcn1(x, edge_index))
        x = F.relu(self.stgcn2(x, edge_index))
        x = self.fc(x)
        return x

# 实例化模型并训练
num_features = 5
hidden_size = 64
Kt = 3
output_size = 1

model = STGCNModel(num_features, hidden_size, Kt, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

        # ...继续上一个代码块

    # 假设已经处理好了输入数据，data_x 是一个 [num_nodes, num_features, Kt] 的张量
    # 假设 data_y 是已经处理好的行驶时间数据，具有与 data_x 相同的形状
    data_y = torch.tensor([...], dtype=torch.float)

    pred = model(data_x, graph.edge_index)
    loss = F.mse_loss(pred, data_y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# 评估模型性能
model.eval()
with torch.no_grad():
    test_data_x = torch.tensor([...], dtype=torch.float)
    test_data_y = torch.tensor([...], dtype=torch.float)

    test_pred = model(test_data_x, graph.edge_index)
    test_loss = F.mse_loss(test_pred, test_data_y)
    print(f'Test Loss: {test_loss.item()}')
