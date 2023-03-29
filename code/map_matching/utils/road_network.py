from pathlib import Path
import os
import networkx as nx

from map_matching.utils.constants import ROAD_DATA_PATH, INTERSEC_DATA_PATH, GRAPH_DATA, ROAD_ATTR, INTERSEC_ATTR
from map_matching.utils.tools import get_road_data, get_intersection_data, get_graph_data


class BaseRoadNetwork:
    def __init__(self, usage):
        self.usage = usage
        self.road_graph = nx.DiGraph()
        self.roads_dict = {}
        self.intersection_dict = {}
        if usage == "match":
            self.init_graph()

    def init_graph(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        work_path = cur_path + "\\.." if "utils" in cur_path else cur_path
        road_data = [get_road_data(road_file) for road_file in
                     Path(os.path.join(work_path, ROAD_DATA_PATH)).iterdir()]  # save road data iterator
        intersection_data = [get_intersection_data(intersection_file) for intersection_file in
                             Path(os.path.join(work_path, INTERSEC_DATA_PATH)).iterdir()]  # save intersection data iterator
        graph_data = [get_graph_data(graph_file) for graph_file in
                      Path(os.path.join(work_path, GRAPH_DATA)).iterdir()]

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
            link_id = int(road_one["link_id"])
            road_one["average_speed"] = float(road_one["average_speed"])
            del road_one["link_id"]
            del road_one["geometry"]
            del road_one["name"]
            if self.usage == "match":
                self.road_graph.add_node(link_id, **road_one)
            else:
                # ['from_node_id', 'to_node_id', 'name', 'length', 'lanes', 'free_speed', 'average_speed']
                # self.road_graph.add_node(link_id, road_attr=tuple([value for value in road_one.values()]))
                self.road_graph.add_node(link_id, **road_one)

        for graph_iter in graph_data:
            for edge in graph_iter:
                if self.usage == "match":
                    self.road_graph.add_edge(edge[0], edge[1], **self.intersection_dict[int(edge[2])])
                else:
                    # self.road_graph.add_edge(edge[0], edge[1], intersec_attr=tuple([
                    #     value for value in self.intersection_dict[edge[2]].values()]))
                    if "name" in self.intersection_dict[int(edge[2])].keys():
                        del self.intersection_dict[int(edge[2])]["name"]
                    self.road_graph.add_edge(edge[0], edge[1], **self.intersection_dict[int(edge[2])])


# a = BaseRoadNetwork("gcn")
# a.init_graph()
#
# print(a.road_graph.nodes[0])
