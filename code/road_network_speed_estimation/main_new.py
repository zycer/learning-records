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
        embed_dim = 3
        self.conv_1 = GCNConv(embed_dim*7, embed_dim*5)
        self.conv_2 = GCNConv(embed_dim*2, embed_dim)
        self.conv_3 = GCNConv(embed_dim, 1)

        # self.item_embedding = torch.nn.Embedding(num_embeddings=99999999, embedding_dim=embed_dim)

        self.lin_1 = torch.nn.Linear(embed_dim, 7*embed_dim)
        self.lin_2 = torch.nn.Linear(embed_dim*5, 2*embed_dim)
        self.lin_3 = torch.nn.Linear(embed_dim, embed_dim)

        self.act = torch.nn.ReLU()

    def forward(self, data):
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(device)

        # 第一层GCN
        x1 = self.lin_1(x)
        x1 = self.act(self.conv_1(x1, edge_index))
        # 第二层GCN
        x2 = self.lin_2(x1)
        x2 = self.act(self.conv_2(x2, edge_index))
        # 第三层GCN
        x3 = self.lin_3(x2)
        x3 = self.act(self.conv_3(x3, edge_index))

        return x3


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




