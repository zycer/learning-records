import torch
import tqdm
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.loader import DataLoader
from road_network_speed_estimation.gnn_estimation.data_process_pyg import RoadNetworkGraphData


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

    def encode(self, x, edge_index):
        x = nn.functional.relu(self.conv1(x, edge_index))
        x = nn.functional.relu(self.conv2(x, edge_index))
        return torch.split(x, self.latent_size, dim=-1)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, edge_index):
        z = nn.functional.relu(self.conv3(z, edge_index))
        z = torch.tanh(self.conv4(z, edge_index))
        return z

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def elbo(z, mu, logvar):
    # 重构误差
    reconstruction_error = torch.nn.functional.binary_cross_entropy_with_logits(z, data, reduction='sum')
    # KL散度
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # ELBO
    elbo_loss = reconstruction_error + kld
    return elbo_loss


if __name__ == '__main__':
    road_graph_data = RoadNetworkGraphData("data/train_data")
    train_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(road_graph_data, batch_size=1, shuffle=False)

    learning_rate = 0.01
    epochs = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    vae = BayesianGCNVAE(3, 256, 128).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        nodes_num = 0
        for data in iter(train_loader):
            print("#", end="")
            data = data.to(device)
            optimizer.zero_grad()
            output = vae(data.x, data.edge_index)
            recon_x, mu, logvar = output
            loss = vae.loss(recon_x, data.x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nodes_num = data.x.shape[0]

        print("\nEpoch:", epoch + 1, "Loss:", total_loss / (len(train_loader.dataset)*nodes_num))
