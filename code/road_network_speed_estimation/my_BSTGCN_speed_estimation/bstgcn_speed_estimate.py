import math
import os
import time

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import FeaStConv
from torch.optim import Adam, RMSprop
from torch.utils.tensorboard import SummaryWriter
from road_network_speed_estimation.utils import BayesianGCNVAE
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import edge_standard_scaler
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import get_st_graph_loader


# 更新BayesianGCNVAE类以接受STGCN输出
class STGCNBayesianGCNVAE(nn.Module):
    def __init__(self, _num_features, _hidden_size, _latent_size, _out_size, _combined_edge_features_dim=9):
        super(STGCNBayesianGCNVAE, self).__init__()
        self.stgcn1 = FeaStConv(_num_features, _hidden_size, 2)
        self.stgcn2 = FeaStConv(_hidden_size, _latent_size, 2)
        self.liner = nn.Linear(_latent_size, _out_size)
        self.relu = nn.ReLU()
        self.bayesian_gcn_vae = BayesianGCNVAE(_num_features, _hidden_size, _latent_size)
        self.edge_time_predictor = nn.Linear(_combined_edge_features_dim, 1)  # 添加一个线性层以预测行驶时间

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


class GATDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=1, concat=True):
        super(GATDiscriminator, self).__init__()
        self.gat_layer1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=concat)
        self.gat_layer2 = GATConv(hidden_channels * num_heads if concat else hidden_channels, hidden_channels,
                                  heads=num_heads, concat=concat)
        self.fc = nn.Linear(hidden_channels * num_heads if concat else hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, _x, _edge_index, _edge_weight):
        _x = nn.functional.relu(self.gat_layer1(_x, _edge_index, _edge_weight))
        _x = nn.functional.relu(self.gat_layer2(_x, _edge_index, _edge_weight))
        _x = self.fc(_x)
        _x = self.sigmoid(_x)
        return _x


def train():
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Loading model successfully.")
    writer = SummaryWriter("run_logs/TTDE train 1")
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
            exit()

        torch.save(model.state_dict(), model_save_path)

    writer.close()


def predict():
    print("Starting model prediction.")
    writer = SummaryWriter("run_logs/TTDE predict 1")
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
                writer.add_scalar("mse", mse, num_batch + num * math.ceil(100 / 3))
                batch_loss_list.append(mse)

        print(f"-data file: {test_data_file}, Loss: {sum(batch_loss_list) / len(batch_loss_list)}\n")
    writer.close()


def gans_train():
    generator = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
    discriminator = GATDiscriminator(out_size, 64, 4).double().to(device)
    # 选择优化器
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # 定义损失函数
    bce_loss = torch.nn.BCELoss()
    reconstruction_loss = nn.MSELoss()  # 用于生成器的重构损失

    for num, train_data_file in enumerate(train_data_files):
        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, train_data_file))
        generator_loss_list = []
        discriminator_loss_list = []

        for epoch in range(num_epochs):
            for batch in tqdm.tqdm(snapshot_graphs_loader):
                snapshot_batch = batch.to(device)
                # 判别器训练
                discriminator_optimizer.zero_grad()
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                edge_index = edge_index[:, :10]
                edge_weight = edge_weight[0:10]
                # 使用生成器生成图
                recon_x, mu, logvar, predicted_edge_time = generator(x.double(), edge_index, edge_weight)
                # 计算判别器在真实数据上的损失
                real_labels = torch.ones(x.size(0), 1).double().to(device)
                real_preds = discriminator(x.double(), edge_index, edge_weight)
                real_loss = bce_loss(real_preds, real_labels)

                # 计算判别器在生成数据上的损失
                fake_labels = torch.zeros(x.size(0), 1).double().to(device)
                print(recon_x.shape)
                print(edge_index.shape)
                print(predicted_edge_time.shape)
                fake_preds = discriminator(recon_x, edge_index, predicted_edge_time)
                fake_loss = bce_loss(fake_preds, fake_labels)

                # 总判别器损失
                discriminator_loss = real_loss + fake_loss
                discriminator_loss_list.append(discriminator_loss.item())
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                # 生成器训练
                generator_optimizer.zero_grad()
                # 计算生成器损失
                generator_fake_preds = discriminator(recon_x, edge_index, predicted_edge_time)
                generator_loss = bce_loss(generator_fake_preds, real_labels)  # 将生成数据误导为真实数据
                recon_loss = reconstruction_loss(x, recon_x)  # 计算重构损失
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算KL散度损失
                generator_total_loss = generator_loss + recon_loss + kl_divergence
                generator_loss_list.append(generator_total_loss.item())

                generator_total_loss.backward()
                generator_optimizer.step()

            # 输出日志
            print(
                f"Epoch {epoch}, Generator loss: {sum(generator_loss_list) / len(generator_loss_list)}, \
                Discriminator loss: {sum(discriminator_loss_list) / len(discriminator_loss_list)}")


if __name__ == '__main__':
    # 超参数
    num_features = 7
    hidden_size = 32
    latent_size = 16
    out_size = 7
    combined_edge_features_dim = 9  # 组合边特征维度

    num_epochs = 10
    learning_rate = 0.01
    alpha = 1.0  # 重构损失权重
    beta = 1.0  # KL散度损失权重
    gamma = 1.0  # 行驶时间预测损失权重

    torch.cuda.empty_cache()

    model_save_path = "saved_models/ttde_model_1.pth"
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

    print("训练集：", train_data_files)
    print("测试集：", test_data_files, end="\n\n")

    # 模型训练
    # train()
    gans_train()

    # 模型预测
    predict()