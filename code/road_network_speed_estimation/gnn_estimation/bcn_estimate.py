import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter
from tqdm import tqdm

from road_network_speed_estimation.gnn_estimation.data_process_pyg import RoadNetworkGraphData

import math



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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BayesianGCNVAE, self).__init__()
        self.conv1 = BayesianGCNConv(in_channels, hidden_channels)
        self.conv2 = BayesianGCNConv(hidden_channels, hidden_channels)
        self.conv3 = BayesianGCNConv(hidden_channels, out_channels)
        self.tanh = nn.Tanh()
        self.kl_loss = 0

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        x = self.conv3(z, edge_index)
        x = self.tanh(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, edge_index):
        mu, logvar = self.conv_mean_var(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar

    def conv_mean_var(self, x, edge_index):
        x = self.encode(x, edge_index)
        mu, logvar = x.mean(dim=0), x.var(dim=0, unbiased=False)
        self.kl_loss = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
        return mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        return recon_loss + self.kl_loss



def train(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0

    for data in data_loader:
        data = data.to(device)

        optimizer.zero_grad()
        recon_data, mu, std = model(data.x, data.edge_index)

        # Compute loss
        recon_loss = F.mse_loss(recon_data, data.x, reduction='mean')
        kl_loss = 0.5 * torch.sum(torch.exp(std) + mu ** 2 - 1 - std)
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(data_loader.dataset)



if __name__ == '__main__':
    road_graph_data = RoadNetworkGraphData()
    data_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)
    gpu_device = "cuda:0"
    device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")
##############################################
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BayesianGCNVAE(128, 256, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, data_loader, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')


