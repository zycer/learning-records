import torch.nn
import torch.nn.functional as F
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from data_process import RoadNetworkGraphData


class GCN(torch.nn.Module):
    def __init__(self):
        embed_dim = 3
        super().__init__()
        self.conv_1 = GCNConv(embed_dim, embed_dim)
        self.pool_1 = TopKPooling(embed_dim, ratio=0.8)

        self.conv_2 = GCNConv(embed_dim, embed_dim)
        self.pool_2 = TopKPooling(embed_dim, ratio=0.8)

        self.conv_3 = GCNConv(embed_dim, embed_dim)
        self.pool_3 = TopKPooling(embed_dim, ratio=0.8)

        # self.item_embedding = torch.nn.Embedding(num_embeddings=99999999, embedding_dim=embed_dim)

        self.lin_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.lin_2 = torch.nn.Linear(embed_dim, embed_dim // 2)
        self.lin_3 = torch.nn.Linear(embed_dim // 2, 1)

        self.bn_1 = torch.nn.BatchNorm1d(embed_dim)
        self.bn_2 = torch.nn.BatchNorm1d(embed_dim // 2)

        self.act_1 = torch.nn.ReLU()
        self.act_2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(x)
        # x = self.item_embedding(x.long())  # 特征编码
        # x = x.squeeze(1)

        x = F.relu(self.conv_1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool_1(x, edge_index, None, batch)
        x_1 = gap(x, batch)

        x = F.relu(self.conv_2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool_2(x, edge_index, None, batch)
        x_2 = gap(x, batch)

        x = F.relu(self.conv_3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool_3(x, edge_index, None, batch)
        x_3 = gap(x, batch)

        x = x_1 + x_2 + x_3

        x = self.lin_1(x)
        x = self.act_1(x)
        x = self.lin_2(x)
        x = self.act_2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin_3(x).squeeze(1))
        return x


def train(model, train_loader):
    model.train()
    criterion = torch.nn.MSELoss()  # 均方损失函数
    optimizer = optim.Adam(params=model.parameters())
    loss_all = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = criterion(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all


if __name__ == '__main__':
    gcn_model = GCN()
    road_graph_data = RoadNetworkGraphData()
    data_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)

    for epoch in range(10):
        loss = train(gcn_model, iter(data_loader))
        print(loss)

