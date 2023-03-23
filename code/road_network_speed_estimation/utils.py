import torch
from sklearn.preprocessing import StandardScaler
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
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class BayesianGCNVAE(nn.Module):
    def __init__(self, num_features, hidden_size, latent_size):
        super(BayesianGCNVAE, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.conv1 = BayesianGCNConv(num_features, hidden_size)
        self.conv2 = BayesianGCNConv(hidden_size, 2 * latent_size)
        self.conv3 = BayesianGCNConv(latent_size, hidden_size)
        self.conv4 = BayesianGCNConv(hidden_size, num_features)

        self.reconstruction_loss = nn.MSELoss(reduction='sum')

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
        return self.decode(z, edge_index, edge_weight), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print(BCE, KLD)
        return BCE + KLD


def z_score(raw_data):
    """
    对原始特征进行z_score标准化
    """
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(raw_data)


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
