import os

import torch
import torch.nn as nn
from torch_geometric.nn.conv import FeaStConv
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from road_network_speed_estimation.utils import BayesianGCNVAE
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import get_st_graph_loader


# 更新BayesianGCNVAE类以接受STGCN输出
class STGCNBayesianGCNVAE(nn.Module):
    def __init__(self, _num_features, _hidden_size, _latent_size, _out_size):
        super(STGCNBayesianGCNVAE, self).__init__()
        self.stgcn1 = FeaStConv(_num_features, _hidden_size, 2)
        self.stgcn2 = FeaStConv(_hidden_size, _latent_size, 2)
        self.liner = nn.Linear(_latent_size, _out_size)
        self.relu = nn.ReLU()
        self.bayesian_gcn_vae = BayesianGCNVAE(_num_features, _hidden_size, _latent_size)

    def forward(self, _x, _edge_index, _edge_weight):
        _x = self.stgcn1(_x, _edge_index)
        _x = self.relu(_x)
        _x = self.stgcn2(_x, _edge_index)
        _x = self.relu(_x)
        _x = self.liner(_x)
        return self.bayesian_gcn_vae(_x, _edge_index, _edge_weight)


def train():
    train_loss_values = []
    writer = SummaryWriter("runs/your_experiment_name")
    for num, train_data_file in enumerate(train_data_files):
        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, train_data_file))
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            epoch_loss_values = []
            for batch in snapshot_graphs_loader:
                snapshot_batch = batch.to(device)
                # 训练模型
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                reconstructed_x, mu, logvar = model(x.double(), edge_index, edge_weight)
                loss = model.bayesian_gcn_vae.loss(reconstructed_x, x.double(), mu, logvar)
                epoch_loss_values.append(loss.item())
                loss.backward()
                print("#", end="")
            optimizer.step()
            epoch_loss = sum(epoch_loss_values) / len(epoch_loss_values)
            writer.add_scalar("Loss/train", epoch_loss, epoch + num * num_epochs)
            print(f"\ndata file: {train_data_file}, Epoch: {epoch + 1}, Loss: {epoch_loss}")

    writer.close()


if __name__ == '__main__':
    num_features = 3
    hidden_size = 32
    latent_size = 16
    out_size = 3
    num_epochs = 10
    learning_rate = 0.01
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # 创建模型、优化器和损失函数
    model = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 分配数据
    data_path = "data"
    data_files = os.listdir(data_path)
    train_ratio = 0.8
    test_ratio = 0.2
    train_data_files = [data_files[index] for index in range(int(len(data_files) * train_ratio))]
    test_data_files = list(set(data_files) - set(train_data_files))

    # 模型训练
    train()
