import os

import torch
import torch.nn as nn
from torch_geometric.nn.conv import FeaStConv
from torch.optim import Adam

from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import get_st_road_graph
from road_network_speed_estimation.utils import BayesianGCNVAE


# 更新BayesianGCNVAE类以接受STGCN输出
class STGCNBayesianGCNVAE(nn.Module):
    def __init__(self, num_features, hidden_size, latent_size):
        super(STGCNBayesianGCNVAE, self).__init__()
        self.stgcn = FeaStConv(num_features, hidden_size, 2)
        self.bayesian_gcn_vae = BayesianGCNVAE(num_features, hidden_size, latent_size)

    def forward(self, _x, _edge_index, _edge_weight):
        _x = self.stgcn(_x, _edge_index)
        return self.bayesian_gcn_vae(x, _edge_index, _edge_weight)


if __name__ == '__main__':
    num_features = 3
    hidden_size = 32
    latent_size = 16
    num_epochs = 10
    learning_rate = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # 创建模型、优化器和损失函数
    model = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size).double().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 加载数据
    data_path = "data"
    data_files = os.listdir(data_path)
    train_ratio = 0.8
    test_ratio = 0.2
    train_data_files = [data_files[index] for index in range(int(len(data_files) * train_ratio))]
    test_data_files = list(set(data_files) - set(train_data_files))

    for train_data_file in train_data_files:
        snapshot_graphs = get_st_road_graph(os.path.join(data_path, train_data_file))
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            loss_list = []
            for index in range(len(snapshot_graphs.graph_attrs)):
                snapshot_data = snapshot_graphs[index]
                snapshot_data = snapshot_data.to(device)
                # 训练模型
                x, edge_index, edge_weight = snapshot_data.x, snapshot_data.edge_index, snapshot_data.edge_attr
                reconstructed_x, mu, logvar = model(x.double(), edge_index, edge_weight)
                loss = model.bayesian_gcn_vae.loss(reconstructed_x, x.double(), mu, logvar)
                loss_list.append(loss.item())
                loss.backward()
                print("#", end="")
            optimizer.step()
            average_loss = sum(loss_list)/len(loss_list)
            print(f"\nEpoch: {epoch + 1}, Loss: {average_loss}")
