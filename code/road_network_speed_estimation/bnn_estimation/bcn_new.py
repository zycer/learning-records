import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
import numpy as np
import math

from road_network_speed_estimation.bnn_estimation.data_process_ms import RoadNetworkGraphData

import mindspore.nn.probability as ms_prob
from mindspore import Tensor


class BayesianGCNConv(nn.Cell):
    """
    Bayesian graph convolutional layer
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        prior_scale (float): Scale of the prior distribution.
        posterior_scale (float): Scale of the posterior distribution.
        dropout_rate (float): Dropout rate for feature map.
    """
    def __init__(self, in_channels, out_channels, prior_scale, posterior_scale, dropout_rate):
        super(BayesianGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.prior_mean = Tensor(np.zeros([out_channels, in_channels]).astype(np.float32))
        self.prior_scale = prior_scale
        self.posterior_mean = ms_prob.Parameter(Tensor(np.zeros([out_channels, in_channels]).astype(np.float32)))
        self.posterior_scale = posterior_scale

        self.weight_prior = ms_prob.Normal(self.prior_mean, self.prior_scale)
        self.weight_posterior = ms_prob.Normal(self.posterior_mean, self.posterior_scale)

        self.bias_prior = ms_prob.Normal(Tensor(np.zeros([out_channels]).astype(np.float32)), self.prior_scale)
        self.bias_posterior = ms_prob.Normal(Tensor(np.zeros([out_channels]).astype(np.float32)), self.posterior_scale)

        self.dropout = nn.Dropout(dropout_rate)

    def _kl_divergence(self):
        """
        Compute the KL divergence between the prior and the posterior distribution.
        """
        weight_kl = ms_prob.kl_divergence(self.weight_posterior, self.weight_prior).sum()
        bias_kl = ms_prob.kl_divergence(self.bias_posterior, self.bias_prior).sum()
        kl_divergence = weight_kl + bias_kl
        return kl_divergence

    def construct(self, x, sample=False):
        """
        Construct BayesianGCNConv
        """
        if sample:
            weight = self.weight_posterior.sample()
            bias = self.bias_posterior.sample()
        else:
            weight = self.weight_posterior.mean()
            bias = self.bias_posterior.mean()

        x = self.dropout(x)
        x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 1), has_bias=True,
                      weight_init=weight, bias_init=bias, pad_mode='same', pad=0, stride=1, dilation=1)(x)

        kl_divergence = self._kl_divergence()
        return x, kl_divergence



class BayesianGCNVAE(nn.Cell):
    """
    Bayesian graph convolutional autoencoder
    Args:
        nfeat (int): Number of input features
        nhid (list): Number of output features for each layer
        dropout_rate (float): Dropout rate for feature map
    """
    def __init__(self, nfeat, nhid, dropout_rate):
        super(BayesianGCNVAE, self).__init__()

        self.nhid = nhid
        self.nout = nout = [nfeat] + nhid + [1]
        self.conv1 = BayesianGCNConv(nout[0], nout[1], 0, 1, dropout_rate)
        self.conv2 = BayesianGCNConv(nout[1], nout[2], 0, 1, dropout_rate)
        self.conv3 = BayesianGCNConv(nout[2], nout[3], 0, 1, dropout_rate)
        self.tanh = nn.Tanh()

    def construct(self, x, sample=False):
        """
        Construct BayesianGCNVAE
        """
        x, kl1 = self.conv1(x, sample)
        x = self.tanh(x)
        x, kl2 = self.conv2(x, sample)
        x = self.tanh(x)
        x, kl3 = self.conv3(x, sample)
        x = self.tanh(x)
        kl = kl1 + kl2 + kl3
        return x, kl


import mindspore.dataset as ds
from mindspore import Tensor
from mindspore import context
from mindspore.train import Model
from mindspore.nn.loss import MSELoss
from mindspore.nn.optim import Adam

# Load data
train_data = RoadNetworkGraphData("./data/train_data", 0, 10).graphs

# Define parameters
input_shape = 4
hidden_sizes = [128, 256, 128]
batch_size = 32
learning_rate = 0.01
num_epochs = 10
dropout_rate = 0.5

# Instantiate model and define optimizer and loss function
model = BayesianGCNVAE(input_shape, hidden_sizes, dropout_rate)
optim = Adam(model.trainable_params(), learning_rate)
loss_fn = MSELoss()

# Train model
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for batch in train_data.create_tuple_iterator():
        batch_data = batch[0]
        target = batch_data
        pred, kl = model(batch_data, sample=True)
        loss = loss_fn(pred, target) + kl
        epoch_loss += loss.asnumpy()
        num_batches += 1
        loss.backward()
        optim.step()
        model.clear_gradients()
    print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / num_batches))
