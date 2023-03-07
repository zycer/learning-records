import os.path as osp
import os
import networkx as nx
import numpy as np

from mindspore.dataset import InMemoryGraphDataset, Graph
from road_network_speed_estimation.utils import z_score, one_graph_node_features_labels


class RoadNetworkGraphData(InMemoryGraphDataset):
    """
    自定义路网图结构
    """

    def __init__(self, data_dir, start=0, end=0):
        self.start = start
        self.end = end
        self.source_graph_data_path = "../source_multi_graph_data"
        super(RoadNetworkGraphData, self).__init__(data_dir=data_dir)

    def raw_paths(self):
        return [osp.join(self.source_graph_data_path,
                         file_name) for file_name in os.listdir(self.source_graph_data_path)[self.start: self.end]]

    def process(self):
        """
        对原始路网图数据进行特征编码、标准化后转换为torch格式数据
        """
        for one_graph_path in self.raw_paths():
            # 有向图转无向图
            road_network_graph = nx.read_graphml(one_graph_path).to_undirected()
            node_features, node_labels = one_graph_node_features_labels(road_network_graph)

            # 道路特征张量化
            node_features_transformed = np.array(z_score(node_features), dtype=np.float64)
            labels_transformed = np.array(z_score(np.array(node_labels).reshape(-1, 1)), dtype=np.float64)

            source_nodes = list(map(lambda x: int(x[0]), road_network_graph.edges))
            target_nodes = list(map(lambda x: int(x[1]), road_network_graph.edges))
            edge_index = np.array([source_nodes, target_nodes], dtype=np.int32)
            graph_data = Graph(edges=edge_index,
                               node_feat={"road_feat": node_features_transformed, "label": labels_transformed})

            self.graphs.append(graph_data)
            print(f"Processed {osp.basename(one_graph_path)}")

        # 持久化处理后的数据
        self.save()


if __name__ == '__main__':
    train_data = RoadNetworkGraphData("./data/train_data", 0, 10)
    test_data = RoadNetworkGraphData("./data/test_data", 10, 15)
    # graphs = road_graph_data.graphs
    #
    # for g in graphs:
    #     nodes = g.get_all_nodes(node_type="0")
    #     print(g.get_node_feature(node_list=nodes, feature_types=["road_feat"]))
    #     exit()
