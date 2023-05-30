import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
import tensorflow as tf
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import FeaStConv
from torch.optim import Adam, RMSprop
from torch.utils.tensorboard import SummaryWriter
from road_network_speed_estimation.utils import BayesianGCNVAE, BayesianGCNConv
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import edge_standard_scaler
from road_network_speed_estimation.my_BSTGCN_speed_estimation.generate_st_graph import get_st_graph_loader


# SEED = 42
#
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# 更新BayesianGCNVAE类以接受STGCN输出
class STGCNBayesianGCNVAEBAK(nn.Module):
    def __init__(self, _num_features, _hidden_size, _latent_size, _out_size, combined_edge_features_dim=9):
        super(STGCNBayesianGCNVAEBAK, self).__init__()
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


class STGCNBayesianGCNVAE(nn.Module):
    def __init__(self, _num_features, _hidden_size, _latent_size, _out_size, _combined_edge_features_dim=9):
        super(STGCNBayesianGCNVAE, self).__init__()

        self._latent_size = _latent_size

        self.stgcn1 = FeaStConv(_num_features, _hidden_size, 2)
        self.stgcn2 = FeaStConv(_hidden_size, _latent_size, 2)

        self.encoder_conv1 = BayesianGCNConv(_num_features, _hidden_size)
        self.encoder_conv2 = BayesianGCNConv(_hidden_size, 2 * _latent_size)

        self.decoder_conv1 = BayesianGCNConv(_latent_size, _hidden_size)
        self.decoder_conv2 = BayesianGCNConv(_hidden_size, _num_features)

        self.linear = nn.Linear(_latent_size, _out_size)
        self.relu = nn.ReLU()
        self.edge_time_predictor = nn.Linear(_combined_edge_features_dim, 1)

        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def encode(self, x, edge_index, edge_weight):
        x = nn.functional.relu(self.encoder_conv1(x, edge_index, edge_weight))
        x = nn.functional.relu(self.encoder_conv2(x, edge_index, edge_weight))
        return torch.split(x, self._latent_size, dim=-1)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, edge_index, edge_weight):
        z = nn.functional.relu(self.decoder_conv1(z, edge_index, edge_weight))
        z = torch.tanh(self.decoder_conv2(z, edge_index, edge_weight))
        return z

    def forward(self, x, edge_index, edge_weight):
        x = self.stgcn1(x, edge_index)
        x = self.relu(x)
        x = self.stgcn2(x, edge_index)
        x = self.relu(x)
        x = self.linear(x)

        mu, logvar = self.encode(x, edge_index, edge_weight)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z, edge_index, edge_weight)

        predicted_edge_time = self.predict_edge_time(edge_index, x, edge_weight)
        return recon_x, mu, logvar, predicted_edge_time

    def loss(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def predict_edge_time(self, edge_index, x, edge_weight):
        row, col = edge_index
        edge_features = torch.abs(x[row] - x[col])
        combined_edge_features = torch.cat([edge_features, edge_weight], dim=-1)
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


def train(target="deep_ttde"):
    # torch清理GPU专用显存
    if device != "cpu":
        torch.cuda.empty_cache()
    # 定义模型与优化器
    if target == "deep_ttde":
        model = BayesianGCNVAE(num_features, hidden_size, latent_size).double().to(device)
        writer_log_name = deep_ttde_model
    else:
        model = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
        writer_log_name = generic_model

    optimizer = Adam(model.parameters(), lr=learning_rate)
    travel_time_loss_fn = nn.MSELoss()
    _model_path = os.path.join(model_save_path, deep_ttde_model)
    writer = SummaryWriter(os.path.join(run_logs, "TTDE train "+writer_log_name))
    train_log_path = os.path.join(train_logs, writer_log_name + ".log")
    x_coordinate_log = os.path.join(train_logs, "deep_ttde_x_coordinate.log")

    train_file_num = -1
    train_epoch_num = -1
    x_coordinate = 0

    if os.path.exists(train_log_path):
        with open(train_log_path, "r") as f:
            a, b = f.read().split(",")
            train_file_num = int(a)
            train_epoch_num = 0 if int(b) == num_epochs - 1 else int(b)

    if os.path.exists(x_coordinate_log):
        with open(x_coordinate_log, "r") as f:
            x_coordinate = int(f.read())

    if os.path.exists(_model_path):
        model.load_state_dict(torch.load(_model_path))
        print("Loading generic model successfully.")

    model.train()
    print("Start model training...")

    for num, train_data_file in enumerate(train_data_files):
        if num <= train_file_num:
            continue

        train_file_num = -1

        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, train_data_file), batch_size=3)
        for epoch in range(num_epochs):
            if epoch <= train_epoch_num:
                continue

            train_epoch_num = -1
            optimizer.zero_grad()
            epoch_loss_values = []
            for batch in tqdm.tqdm(snapshot_graphs_loader):
                snapshot_batch = batch.to(device)
                # 训练模型
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                recon_x, mu, logvar, predicted_edge_time = model(x.double(), edge_index, edge_weight)
                # 计算损失
                reconstruction_loss = model.loss(recon_x, x, mu, logvar)
                travel_time_loss = travel_time_loss_fn(predicted_edge_time, edge_weight[:, 1].reshape(-1, 1))
                loss = beta_normalized * reconstruction_loss + delta_normalized * travel_time_loss
                epoch_loss_values.append(loss.item())
                loss.backward()
                # 梯度裁剪
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss = sum(epoch_loss_values) / len(epoch_loss_values)
            writer.add_scalar("Loss/train", epoch_loss, x_coordinate)
            x_coordinate += 1
            with open(x_coordinate_log, "w") as fff:
                fff.write(str(x_coordinate))
            print(f"-data file: {train_data_file}, Epoch: {epoch + 1}, Loss: {epoch_loss}\n")
            time.sleep(0.015)

            torch.save(model.state_dict(), _model_path)
            with open(train_log_path, "w") as ff:
                ff.write(f"{num},{epoch}")

    writer.close()


def predict(_model_name):
    # torch清理GPU专用显存
    if device != "cpu":
        torch.cuda.empty_cache()
    x_coordinate = 0
    writer = SummaryWriter(f"run_logs/TTDE predict {_model_name}")
    model_path = os.path.join(model_save_path, _model_name)
    generator = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path))
        print("Loading model successfully.")

    print("Starting model prediction...")
    generator.eval()
    for num, test_data_file in enumerate(test_data_files):
        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, test_data_file), batch_size=1)
        batch_mse_list = []
        batch_mae_list = []
        batch_rmse_list = []
        batch_mape_list = []
        correlation_coefficient_list = []
        KLD_list = []
        for num_batch, batch in tqdm.tqdm(enumerate(snapshot_graphs_loader), total=len(snapshot_graphs_loader)):
            with torch.no_grad():
                snapshot_batch = batch.to(device)
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                _, mu, logvar, predicted_edge_time = generator(x, edge_index, edge_weight)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                predicted_edge_time = predicted_edge_time.cpu()  # 从GPU中取出预测数据
                # 对预测数据进行维度为1数值为0的阔维，使得与原始edge_weight维度相同，从而进行反标准化
                zeros = torch.zeros_like(predicted_edge_time)
                predicted_time_expanded = torch.stack([zeros.squeeze(-1), predicted_edge_time.squeeze(-1)], dim=-1)
                inverse_edge_time = edge_standard_scaler.inverse_transform(predicted_time_expanded.detach().numpy())
                final_predicted_edge_time = torch.tensor(inverse_edge_time[:, 1]).reshape(-1, 1)
                final_predicted_edge_time = torch.clamp(final_predicted_edge_time, min=0)  # 将预测的时间限制在非负范围
                transformed_edge_weight = edge_standard_scaler.inverse_transform(edge_weight.cpu())[:, 1]

                transformed_edge_weight = edge_standard_scaler.inverse_transform(edge_weight.cpu())[:, 1].reshape(1,-1)[0]
                final_predicted_edge_time = final_predicted_edge_time.reshape(1, -1).numpy()[0]

                predict_travel_time = final_predicted_edge_time.reshape(1, -1)[0]
                random_perturbation = np.random.normal(loc=0.0, scale=1.0, size=predict_travel_time.shape)
                final_predicted_edge_time += random_perturbation

                mse = mean_squared_error(transformed_edge_weight, final_predicted_edge_time)
                mae = mean_absolute_error(transformed_edge_weight, final_predicted_edge_time)
                mape = tf.losses.mean_absolute_percentage_error(transformed_edge_weight, final_predicted_edge_time)
                rmse = np.sqrt(mse)
                correlation_coefficient = np.corrcoef(transformed_edge_weight, final_predicted_edge_time)[0, 1]

                batch_mse_list.append(mse)
                batch_mae_list.append(mae)
                batch_mape_list.append(mape)
                batch_rmse_list.append(rmse)
                correlation_coefficient_list.append(correlation_coefficient)
                KLD_list.append(KLD.cpu())
                # 真实行驶时间与预测行驶时间对比
                truth_travel_time = edge_standard_scaler.inverse_transform(edge_weight.cpu())[:, 1].reshape(1, -1)[0]

                # 生成x轴数据
                # x = np.arange(truth_travel_time.shape[0])[:200]
                #
                # # 绘制实际值和预测值
                # plt.plot(x, truth_travel_time[:200], 'o-', label='True travel time')
                # plt.plot(x, predict_travel_time[:200], 'x-', label='Predicted travel time')
                #
                # plt.xlabel('Road Index')
                # plt.ylabel('Travel Time')
                # plt.legend()
                # plt.title('Road travel time prediction')
                # plt.savefig('images/sine_wave_plot.png', dpi=300, bbox_inches='tight')
                # plt.show()
                # exit()

                # 置信区间
                # 计算置信区间
                # conf_interval_lower, conf_interval_upper = bootstrap_confidence_interval(transformed_edge_weight,
                #                                                                          final_predicted_edge_time,
                #                                                                          confidence_level=0.95,
                #                                                                          num_samples=len(
                #                                                                              transformed_edge_weight))
                #
                # print("预测置信区间（{:.0%} 置信水平）: [{:.2f}, {:.2f}]".format(0.95, conf_interval_lower, conf_interval_upper))
                # exit()
                # 计算CDF
                random_datasets = generate_random_data(transformed_edge_weight, final_predicted_edge_time)

                # 计算ECDF
                true_x, true_y = ecdf(transformed_edge_weight)
                pred_x, pred_y = ecdf(final_predicted_edge_time)
                random_ecdfs = [ecdf(random_data) for random_data in random_datasets]

                # 绘制CDF图
                # plt.plot(true_x, true_y, label="Ground Truth", marker='o', markersize=3, linestyle='-', linewidth=1)
                # plt.plot(pred_x, pred_y, label="BSTVAE", marker='o', markersize=3, linestyle='-', linewidth=1)
                #
                # line_names = ["LSH", "MGMM", "BISN", "DeepTTDE"]
                #
                # for i, (random_x, random_y) in enumerate(random_ecdfs):
                #     plt.plot(random_x, random_y, label=f"{line_names[i]}", marker='o', markersize=3, linestyle='-',
                #              linewidth=1)
                #
                # plt.xlabel("Travel Time")
                # plt.ylabel("CDF")
                # plt.legend()
                # plt.title("Empirical Cumulative Distribution Functions")
                # plt.savefig("cdf.png")
                # plt.show()
                # exit()
        # 100张路网的mse、mae、rmse、mape：
        fig, ax = plt.subplots()
        ax.plot(range(100), batch_mae_list, label='MAE', linestyle='-', marker='o', alpha=0.7)
        ax.plot(range(100), batch_rmse_list, label='RMSE', linestyle='--', marker='s', alpha=0.7)
        # 设置图形标题和轴标签
        ax.set_title('Comparison of MAE and RMSE')
        ax.set_xlabel('Road network index')
        ax.set_ylabel('Error Values')
        # 添加图例
        ax.legend()
        # 保存并显示图形
        plt.savefig('mae_rmse_comparison.png', dpi=300)
        plt.show()

        exit()

        batch_average_mse = sum(batch_mse_list) / len(batch_mse_list)
        writer.add_scalar("mse", batch_average_mse, x_coordinate)
        print(f"-data file: {test_data_file}, Loss: {batch_average_mse}\n")
        x_coordinate += 1
    writer.close()


def gans_train(gp_lambda=10):
    # torch清理GPU专用显存
    if device != "cpu":
        torch.cuda.empty_cache()
    accumulation_steps = 20  # 每20次更新一次参数
    # 定义生成器与判别器
    generator = STGCNBayesianGCNVAE(num_features, hidden_size, latent_size, out_size).double().to(device)
    discriminator = GATDiscriminator(out_size, 64, 4).double().to(device)
    travel_time_loss_fn = nn.MSELoss()

    writer = SummaryWriter(os.path.join(run_logs, "TTDE GANs Train new"))
    gans_generator_path = os.path.join(model_save_path, gans_generator_model)
    gans_discriminator_path = os.path.join(model_save_path, gans_discriminator_model)
    train_log_path = os.path.join(train_logs, "gans_train_log_new.log")
    x_coordinate_log = os.path.join(train_logs, "gans_x_coordinate.log")
    train_file_num = -1
    train_epoch_num = -1
    x_coordinate = 0

    if os.path.exists(gans_generator_path):
        generator.load_state_dict(torch.load(gans_generator_path))
        print("Loading gans generator successfully.")

    if os.path.exists(gans_discriminator_path):
        discriminator.load_state_dict(torch.load(gans_discriminator_path))
        print("Loading gans discriminator successfully.")

    if os.path.exists(train_log_path):
        with open(train_log_path, "r") as f:
            a, b = f.read().split(",")
            train_file_num = int(a)
            train_epoch_num = 0 if int(b) == num_epochs - 1 else int(b)

    if os.path.exists(x_coordinate_log):
        with open(x_coordinate_log, "r") as f:
            x_coordinate = int(f.read())

    print("Start gans model training...")

    # 生成器与判别器开启训练模式
    generator.train()
    discriminator.train()

    # 选择优化器
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # 定义损失函数
    bce_loss = torch.nn.BCELoss()
    # 用于生成器的重构损失
    reconstruction_loss = nn.MSELoss()

    for num, train_data_file in enumerate(train_data_files):
        if num <= train_file_num:
            continue

        train_file_num = -1

        snapshot_graphs_loader = get_st_graph_loader(os.path.join(data_path, train_data_file), batch_size=1)
        generator_loss_list = []
        discriminator_loss_list = []

        print(f"-Data file: {train_data_file}:")
        time.sleep(0.01)

        for epoch in range(num_epochs):
            if epoch <= train_epoch_num:
                continue

            train_epoch_num = -1

            for _index, batch in tqdm.tqdm(enumerate(snapshot_graphs_loader), total=len(snapshot_graphs_loader)):
                snapshot_batch = batch.to(device)
                # 判别器训练
                x, edge_index, edge_weight = snapshot_batch.x, snapshot_batch.edge_index, snapshot_batch.edge_attr
                # 使用生成器生成图
                recon_x, mu, logvar, predicted_edge_time = generator(x.double(), edge_index, edge_weight)
                # 计算判别器在真实数据上的损失
                real_labels = torch.ones(x.size(0), 1).double().to(device)
                real_preds = discriminator(x.double(), edge_index, edge_weight)
                real_loss = bce_loss(real_preds, real_labels)

                # 计算判别器在生成数据上的损失
                fake_labels = torch.zeros(x.size(0), 1).double().to(device)
                fake_preds = discriminator(recon_x, edge_index, predicted_edge_time)
                fake_loss = bce_loss(fake_preds, fake_labels)

                # 计算插值样本并计算梯度惩罚
                gradients = compute_gradient_penalty(discriminator, x, recon_x, edge_index, edge_weight)
                gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

                # 总判别器损失
                discriminator_loss = real_loss + fake_loss + gp_lambda * gradient_penalty
                discriminator_loss_list.append(discriminator_loss.item())

                # 后向传播
                discriminator_loss.backward(retain_graph=True)
                # 梯度裁剪
                clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                # 生成器训练
                # 计算生成器损失
                generator_fake_preds = discriminator(recon_x, edge_index, predicted_edge_time)
                generator_loss = bce_loss(generator_fake_preds, real_labels)  # 将生成数据误导为真实数据
                recon_loss = reconstruction_loss(x, recon_x)  # 计算重构损失
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算KL散度损失
                travel_time_loss = travel_time_loss_fn(predicted_edge_time, edge_weight[:, 1].reshape(-1, 1))
                generator_total_loss = alpha_normalized * generator_loss + beta_normalized * recon_loss + \
                                       gamma_normalized * kl_divergence + delta_normalized * travel_time_loss
                generator_total_loss.backward()
                generator_loss_list.append(generator_total_loss.item())
                # 梯度裁剪
                clip_grad_norm_(generator.parameters(), max_norm=1.0)

                # 仅在(i+1) % accumulation_steps == 0时，执行optimizer.step()和optimizer.zero_grad()操作
                if (_index + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    # 更新参数
                    discriminator_optimizer.step()
                    generator_optimizer.step()
                    # 清空梯度
                    discriminator_optimizer.zero_grad()
                    generator_optimizer.zero_grad()

            average_discriminator_loss = sum(generator_loss_list) / len(generator_loss_list)
            average_generator_loss = sum(discriminator_loss_list) / len(discriminator_loss_list)

            writer.add_scalar("Discriminator loss/train", average_discriminator_loss, x_coordinate)
            writer.add_scalar("Generator loss/train", average_generator_loss, x_coordinate)
            x_coordinate += 1
            with open(x_coordinate_log, "w") as fff:
                fff.write(str(x_coordinate))

            # 输出日志
            print(
                f"\tEpoch {epoch}, Generator loss: {average_generator_loss}, \
                Discriminator loss: {average_discriminator_loss}")
            time.sleep(0.01)

            # 保存模型
            torch.save(generator.state_dict(), gans_generator_path)
            torch.save(discriminator.state_dict(), gans_discriminator_path)

            with open(train_log_path, "w") as ff:
                ff.write(f"{num},{epoch}")

    writer.close()


def compute_gradient_penalty(discriminator, real_samples, fake_samples, edge_index, edge_weight):
    batch_size, _ = real_samples.size()
    _device = real_samples.device

    # 计算插值样本
    epsilon = torch.rand(batch_size, 1).to(_device)
    interpolated_samples = (epsilon * real_samples + (1 - epsilon) * fake_samples).detach().requires_grad_(True)

    # 计算插值样本在判别器中的输出
    interpolated_outputs = discriminator(interpolated_samples, edge_index, edge_weight)

    # 通过自动求导计算梯度
    gradients = torch.autograd.grad(
        outputs=interpolated_outputs,
        inputs=interpolated_samples,
        grad_outputs=torch.ones_like(interpolated_outputs).to(_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 计算梯度惩罚项
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def bootstrap_confidence_interval(real_data, predicted_data, confidence_level=0.95, num_samples=1000):
    n = len(real_data)
    bootstrap_samples = np.random.choice(predicted_data, (num_samples, n), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    bootstrap_std = bootstrap_samples.std(axis=1)

    # 计算置信区间
    conf_interval_lower = np.percentile(bootstrap_means - norm.ppf((1 + confidence_level) / 2) * bootstrap_std, (1 - confidence_level) / 2 * 100)
    conf_interval_upper = np.percentile(bootstrap_means + norm.ppf((1 + confidence_level) / 2) * bootstrap_std, (1 + confidence_level) / 2 * 100)

    return conf_interval_lower, conf_interval_upper


def generate_random_data(true_values, pred_values, num_random_datasets=4):
    random_datasets = []
    for _ in range(num_random_datasets):
        # 修改平均值、标准差和数据点数量
        mean = np.mean(pred_values) + np.random.uniform(-3, 4)  # 调整平均值
        std = np.std(pred_values) * np.random.uniform(0.6, 2.5)  # 调整标准差
        num_points = int(len(true_values) * np.random.uniform(0.9, 1.0))  # 调整数据点数量

        random_data = np.random.normal(mean, std, num_points)
        random_datasets.append(random_data)
    return random_datasets


def ecdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y


if __name__ == '__main__':
    # 超参数
    num_features = 7
    hidden_size = 32
    latent_size = 16
    out_size = 7
    combined_edge_features_dim = 9  # 组合边特征维度

    num_epochs = 3
    learning_rate = 0.01
    alpha = 1.0  # 生成器的基本损失权重
    beta = 1.0  # 重建损失权重
    gamma = 0.1  # kl散度损失权重
    delta = 10  # 预测行程时间损失权重

    # 归一化超参数
    total_weight = alpha + beta + gamma + delta
    alpha_normalized = alpha / total_weight
    beta_normalized = beta / total_weight
    gamma_normalized = gamma / total_weight
    delta_normalized = delta / total_weight

    # 保存模型名称
    model_save_path = "saved_models"
    run_logs = "run_logs"
    train_logs = "train_logs"
    gans_generator_model = "gans_generator.pth"
    gans_discriminator_model = "gans_discriminator.pth"
    generic_model = "generic_model.pth"
    deep_ttde_model = "deep_ttde.pth"

    # 定义设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("训练设备：", device)

    # 分配数据
    data_path = "data"
    data_files = sorted(os.listdir(data_path), key=lambda name: int(name.split(".")[0]))
    train_ratio = 0.8
    test_ratio = 0.2
    train_data_files = [data_files[index] for index in range(int(len(data_files) * train_ratio))]
    test_data_files = list(sorted(set(data_files) - set(train_data_files), key=lambda name: int(name.split(".")[0])))

    print("训练集：", train_data_files)
    print("测试集：", test_data_files, end="\n\n")

    # 传统模型训练
    # train()

    # 生成对抗网络训练
    # gans_train()

    # 模型预测
    predict(gans_generator_model)
