import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from torch import nn
from torch_geometric.nn import MessagePassing


class BayesianGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BayesianGCNConv, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_mu = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_rho = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = torch.nn.Parameter(torch.Tensor(out_channels))
        self.edge_feature_combiner = torch.nn.Sequential(
            torch.nn.Linear(in_channels + 2, in_channels),  # 线性层将输入维度从 in_channels + 2 转换为 in_channels
            torch.nn.ReLU()  # 应用ReLU激活函数
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_mu)
        torch.nn.init.constant_(self.weight_rho, -5)
        torch.nn.init.zeros_(self.bias_mu)
        torch.nn.init.constant_(self.bias_rho, -5)

    def forward(self, x, edge_index, edge_weight=None):
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.training:
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + weight_epsilon * self.weight_sigma
            bias = self.bias_mu + bias_epsilon * self.bias_sigma
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        out = torch.matmul(out, weight) + bias

        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        else:
            combined_features = torch.cat([x_j, edge_weight], dim=-1)  # 将边特征与节点特征连接起来
            return self.edge_feature_combiner(combined_features)  # 通过神经网络层处理组合特征


class BayesianGCNVAE(nn.Module):
    def __init__(self, num_features, hidden_size, latent_size, _combined_edge_features_dim=9):
        super(BayesianGCNVAE, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = BayesianGCNConv(num_features, hidden_size)
        self.conv2 = BayesianGCNConv(hidden_size, 2 * latent_size)
        self.conv3 = BayesianGCNConv(latent_size, hidden_size)
        self.conv4 = BayesianGCNConv(hidden_size, num_features)

        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        self.edge_time_predictor = nn.Linear(_combined_edge_features_dim, 1)

    def encode(self, x, edge_index, edge_weight):
        x = nn.functional.relu(self.conv1(x, edge_index, edge_weight))
        x = nn.functional.relu(self.conv2(x, edge_index, edge_weight))
        return torch.split(x, self.latent_size, dim=-1)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, edge_index, edge_weight):
        z = nn.functional.relu(self.conv3(z, edge_index, edge_weight))
        z = torch.tanh(self.conv4(z, edge_index, edge_weight))
        return z

    def forward(self, x, edge_index, edge_weight):
        mu, logvar = self.encode(x, edge_index, edge_weight)
        z = self.reparametrize(mu, logvar)
        predicted_edge_time = self.predict_edge_time(edge_index, x, edge_weight)
        return self.decode(z, edge_index, edge_weight), mu, logvar, predicted_edge_time

    def loss(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def predict_edge_time(self, edge_index, x, edge_weight):
        row, col = edge_index
        edge_features = torch.abs(x[row] - x[col])
        combined_edge_features = torch.cat([edge_features, edge_weight], dim=-1)
        return self.edge_time_predictor(combined_edge_features)


def z_score(raw_data):
    """
    对原始特征进行z_score标准化
    """
    standard_scaler = StandardScaler()
    return torch.tensor(standard_scaler.fit_transform(raw_data), dtype=torch.double)


def min_max_scaler(data):
    scaler = MinMaxScaler()
    # 使用 MinMaxScaler 对数据进行缩放
    return torch.tensor(scaler.fit_transform(data), dtype=torch.double)


def one_graph_node_features_labels(road_network_graph):
    node_features = []
    labels = []
    for node in road_network_graph.nodes:
        node_attr = road_network_graph.nodes[node]
        labels.append(node_attr.pop("average_speed"))
        del node_attr["from_node_id"]
        del node_attr["to_node_id"]
        node_attr = list(node_attr.values())
        node_features.append(node_attr)

    return node_features, labels


def visualize_graph(graph, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()
