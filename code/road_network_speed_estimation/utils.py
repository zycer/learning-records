from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx


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