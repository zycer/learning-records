import os
import time

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import mean_squared_error
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn.conv import FeaStConv
from torch.optim import Adam, RMSprop
from torch.utils.tensorboard import SummaryWriter
from road_network_speed_estimation.utils import BayesianGCNVAE
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import edge_standard_scaler
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import get_st_graph_loader


# 更新BayesianGCNVAE类以接受STGCN输出
class STGCNBayesianGCNVAE(nn.Module):
    def __init__(self, _num_features, _hidden_size, _latent_size, _out_size, combined_edge_features_dim=9):
        super(STGCNBayesianGCNVAE, self).__init__()
        self.stgcn1 = FeaStConv(_num_features, _hidden_size, 2)
        self.stgcn2 = FeaStConv(_hidden_size, _latent_size, 2)
        self.liner = nn.Linear(_latent_size, _out_size)
        self.relu = nn.ReLU()
        self.bayesian_gcn_vae = BayesianGCNVAE(_num_features, _hidden_size, _latent_size)
        self.edge_time_predictor = nn.Linear(combined_edge_features_dim, 1)  # 添加一个线性层以预测行驶时间

    def forward(self, _x, _edge_index, _edge_weight):
        _x = self.stgcn1(_x, _edge_index)
        _x = self.relu(_x)
        _x = self.stgcn2(_x, _edge_index)
        _x = self.relu(_x)
        _x = self.liner(_x)
        _recon_x, _mu, _logvar = self.bayesian_gcn_vae(_x, _edge_index, _edge_weight)
        _predicted_edge_time = self.predict_edge_time(_edge_index, _x, _edge_weight)
        return _recon_x, _mu, _logvar, _predicted_edge_time

    def predict_edge_time(self, edge_index, _x, edge_weight):
        row, col = edge_index
        edge_features = torch.abs(_x[row] - _x[col])  # 使用L1距离作为节点特征之间的差异
        combined_edge_features = torch.cat([edge_features, edge_weight], dim=-1)  # 将原始行驶时间与节点特征差异拼接在一起
        return self.edge_time_predictor(combined_edge_features)


def train():
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Loading model successfully.")
    writer = SummaryWriter("run_logs/TTDE train")
    model.train()
    print("Start model training.")
    for num, train_data_file in enumerate(train_data_files):
        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, train_data_file))
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            epoch_loss_values = []
            for batch in tqdm.tqdm(snapshot_graphs_loader):
                snapshot_batch = batch.to(device)
                # 训练模型
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                recon_x, mu, logvar, predicted_edge_time = model(x.double(), edge_index, edge_weight)
                # 计算损失
                reconstruction_loss = model.bayesian_gcn_vae.loss(recon_x, x, mu, logvar)
                travel_time_loss = travel_time_loss_fn(predicted_edge_time, edge_weight[:, 1].reshape(-1, 1))
                loss = alpha * reconstruction_loss + beta * travel_time_loss + gamma * travel_time_loss
                epoch_loss_values.append(loss.item())
                loss.backward()
                # 梯度裁剪
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss = sum(epoch_loss_values) / len(epoch_loss_values)
            writer.add_scalar("Loss/train", epoch_loss, epoch + num * num_epochs)
            print(f"-data file: {train_data_file}, Epoch: {epoch + 1}, Loss: {epoch_loss}\n")
            time.sleep(0.015)

        torch.save(model.state_dict(), model_save_path)

    writer.close()


def predict():
    print("Starting model prediction.")
    writer = SummaryWriter("run_logs/TTDE predict")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    model.eval()
    for num, test_data_file in enumerate(test_data_files):
        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, test_data_file))
        batch_loss_list = []
        for num_batch, batch in tqdm.tqdm(enumerate(snapshot_graphs_loader), total=len(snapshot_graphs_loader)):
            with torch.no_grad():
                snapshot_batch = batch.to(device)
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                _, _, _, predicted_edge_time = model(x, edge_index, edge_weight)
                predicted_edge_time = predicted_edge_time.cpu()  # 从GPU中取出预测数据
                # 对预测数据进行维度为1数值为0的阔维，使得与原始edge_weight维度相同，从而进行反标准化
                zeros = torch.zeros_like(predicted_edge_time)
                predicted_time_expanded = torch.stack([zeros.squeeze(-1), predicted_edge_time.squeeze(-1)], dim=-1)
                inverse_edge_time = edge_standard_scaler.inverse_transform(predicted_time_expanded.detach().numpy())
                final_predicted_edge_time = torch.tensor(inverse_edge_time[:, 1]).reshape(-1, 1)
                mse = mean_squared_error(edge_standard_scaler.inverse_transform(edge_weight.cpu())[:, 1],
                                         final_predicted_edge_time)
                writer.add_scalar("mse", mse, num_batch + num * num_epochs)
                batch_loss_list.append(mse)

        print(f"-data file: {test_data_file}, Loss: {sum(batch_loss_list) / len(batch_loss_list)}\n")
    writer.close()


if __name__ == '__main__':
    # 超参数
    num_features = 7
    hidden_size = 32
    latent_size = 16
    out_size = 7
    num_epochs = 10
    learning_rate = 0.01
    alpha = 1.0  # 重构损失权重
    beta = 1.0  # KL散度损失权重
    gamma = 1.0  # 行驶时间预测损失权重

    model_save_path = "saved_models/ttde_model.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # 创建模型、优化器
    model = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
    # optimizer = RMSprop(model.parameters(), lr=learning_rate)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    travel_time_loss_fn = nn.MSELoss()

    # 分配数据
    data_path = "data"
    data_files = os.listdir(data_path)
    train_ratio = 0.8
    test_ratio = 0.2
    train_data_files = [data_files[index] for index in range(int(len(data_files) * train_ratio))]
    test_data_files = list(set(data_files) - set(train_data_files))

    # 模型训练
    train()

    # 模型预测
    predict()
