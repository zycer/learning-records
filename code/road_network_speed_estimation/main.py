import numpy as np
import torch.nn
import torch.nn.functional as F
from torch.nn import Upsample
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from data_process import RoadNetworkGraphData
from matplotlib import pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        scale_factor = 15
        embed_dim = scale_factor * 3
        self.up_sample = Upsample(scale_factor=scale_factor, mode="nearest")
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
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.up_sample(torch.tensor(np.array([x.cpu().detach().numpy()])))[0]
        x = x.to(device)
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
        # x = x.squeeze(-1)
        return x


def train(model, train_data):
    model.train()
    criterion = torch.nn.MSELoss()  # 均方损失函数
    optimizer = optim.Adam(params=model.parameters())

    optimizer.zero_grad()
    output = model(train_data)
    label = train_data.y
    _loss = criterion(output, label)
    _loss.backward()
    optimizer.step()
    return _loss


if __name__ == '__main__':
    gpu_device = "cuda:0"
    epoch_size = 12
    device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")
    gcn_model = GCN()
    gcn_model = gcn_model.to(device)
    road_graph_data = RoadNetworkGraphData()
    data_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)

    train_loss_record = []

    for num, one_road_network_data in enumerate(iter(data_loader)):
        loss = 0
        print(f"根据{num+1}张路网数据训练参数...")
        for epoch in range(epoch_size):
            loss += train(gcn_model, one_road_network_data)

        average_loss = loss.cpu().detach().numpy()
        train_loss_record.append(average_loss)
        print("loss:", average_loss)

        plt.plot(train_loss_record, label="training loss")
        plt.legend()
        plt.show()




