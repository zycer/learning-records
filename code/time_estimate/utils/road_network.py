from pathlib import Path

import networkx as nx

from .constants import ROAD_DATA_PATH, INTERSEC_DATA_PATH, GRAPH_DATA, ROAD_ATTR, INTERSEC_ATTR
from .tools import DBManager, get_road_data, get_intersection_data, get_graph_data


class BaseRoadNetwork:
    def __init__(self, usage):
        self.usage = usage
        self.road_graph = nx.DiGraph()
        self.roads_dict = {}
        self.intersection_dict = {}
        self.__init_graph()

    def __init_graph(self):
        road_data = [get_road_data(road_file) for road_file in
                     Path(ROAD_DATA_PATH).iterdir()]  # save road data iterator
        intersection_data = [get_intersection_data(intersection_file) for intersection_file in
                             Path(INTERSEC_DATA_PATH).iterdir()]  # save intersection data iterator
        graph_data = [get_graph_data(graph_file) for graph_file in
                      Path(GRAPH_DATA).iterdir()]

        for road_iter in road_data:
            for road_one_info in road_iter:
                road_attr_dict = dict(zip(ROAD_ATTR, road_one_info))
                self.roads_dict[road_attr_dict["link_id"]] = road_attr_dict
                # traffic_graph.add_edge(road_attr_dict["from_node_id"], road_attr_dict["to_node_id"], **road_attr_dict)

        for intersec_iter in intersection_data:
            for intersec_one_info in intersec_iter:
                intersec_attr_dict = dict(zip(INTERSEC_ATTR, intersec_one_info))
                self.intersection_dict[intersec_attr_dict["node_id"]] = intersec_attr_dict
                # traffic_graph.add_node(intersec_attr_dict["node_id"], **intersec_attr_dict)

        for road_one in self.roads_dict.values():
            link_id = road_one["link_id"]
            del road_one["link_id"]
            if self.usage == "match":
                self.road_graph.add_node(link_id, **road_one)
            else:
                self.road_graph.add_node(link_id, road_attr=tuple([value for value in road_one.values()]))

        for graph_iter in graph_data:
            for edge in graph_iter:
                if self.usage == "match":
                    self.road_graph.add_edge(edge[0], edge[1], **self.intersection_dict[int(edge[2])])
                else:
                    self.road_graph.add_edge(edge[0], edge[1], intersec_attr=tuple([
                        value for value in self.intersection_dict[edge[2]].values()]))
