import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from road_network_speed_estimation.gnn_estimation.data_process_pyg import RoadNetworkGraphData

from torch_geometric.nn import GCNConv, GAE, VGAE


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, hidden_channels)
        self.conv_logstd = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        std = torch.exp(logstd)
        z = mu + std * torch.randn_like(std)
        return z, mu, logstd


class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, z, edge_index):
        x = F.relu(self.conv1(z, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))
        return x


class VAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, out_channels)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, edge_index):
        z, mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        x = self.decoder(z, edge_index)
        return x, mu, logstd


def loss_function(recon_x, x, mu, logstd):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())
    return BCE + KLD


if __name__ == '__main__':

    road_graph_data = RoadNetworkGraphData()
    train_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)
    device = torch.device('cpu' if torch.cuda.is_available() else "cpu")
    num_epochs = 10

    # Initialize model and optimizer
    model = VAE(in_channels=3, hidden_channels=128, out_channels=3)
    optimizer = Adam(model.parameters(), lr=0.01)

    # Train model
    for epoch in range(10):
        for data in train_loader:
            optimizer.zero_grad()
            x, edge_index, _ = data.x, data.edge_index, data.batch
            recon_x, mu, logstd = model(x, edge_index)
            loss = loss_function(recon_x, x, mu, logstd)
            print(loss)
