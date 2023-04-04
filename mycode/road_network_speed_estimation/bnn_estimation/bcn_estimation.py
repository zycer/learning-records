import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
from mindspore.nn import TrainOneStepCell
from mindspore.nn.probability import bnn_layers
import mindspore.ops as ops
from mindspore import context
from data_process_ms import RoadNetworkGraphData


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """conv层的权重初始值"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """fc层的权重初始值"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """初始权重"""
    return TruncatedNormal(0.02)


class BNN(nn.Cell):
    def __init__(self):
        super(BNN, self).__init__()
        embed_dim = 3
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d()

        self.conv1 = bnn_layers.ConvReparam(8, 16, 1)
        self.conv2 = bnn_layers.ConvReparam(32, 16, 1)
        self.conv3 = bnn_layers.ConvReparam(8, 1, 1)

        self.fc1 = bnn_layers.DenseReparam(embed_dim, 8)
        self.fc2 = bnn_layers.DenseReparam(16, 32)
        self.fc3 = bnn_layers.DenseReparam(16, 8)

    def construct(self, x):
        x = x[0]
        x = self.fc1(x)
        x = self.conv1(Tensor([[x]]))
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(self.conv2(x))

        x = self.fc3(x)
        x = self.relu(self.conv3(x))

        return x


def train_model(model_net, net, graphs):
    print("train_model...")
    accs = []
    loss_sum = 0

    for one_graph in graphs:
        all_nodes = one_graph.get_all_nodes(node_type="0")
        features = one_graph.get_node_feature(node_list=all_nodes, feature_types=["road_feat"])
        labels = one_graph.get_node_feature(node_list=all_nodes, feature_types=["label"])

        transformed_features = Tensor(features)
        transformed_labels = Tensor(labels)

        loss = model_net(transformed_features, transformed_labels)
        output = net(transformed_features)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == labels)
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        train_x = Tensor(data['road_data'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        output = net(train_x)
        log_output = ops.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean


if __name__ == "__main__":
    bnn_model = BNN()

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.AdamWeightDecay(params=bnn_model.trainable_params(), learning_rate=0.0001)

    net_with_loss = bnn_layers.WithBNNLossCell(bnn_model, criterion, 60000, 0.000001)
    train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
    train_bnn_network.set_train()

    train_data = RoadNetworkGraphData("./data/train_data", 0, 10).graphs
    test_data = RoadNetworkGraphData("./data/test_data", 10, 15).graphs

    epoch = 1

    for i in range(epoch):
        train_loss, train_acc = train_model(train_bnn_network, bnn_model, train_data)

        # valid_acc = validate_model(bnn_model, test_data)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f}'.format(
            i, train_loss, train_acc))
