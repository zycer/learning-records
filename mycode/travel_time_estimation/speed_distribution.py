import copy
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from constants import ROAD_DATA_PATH, INTERSEC_DATA_PATH, ROAD_ATTR, INTERSEC_ATTR, GRAPH_DATA
from utils import get_road_data, get_intersection_data, get_graph_data
from db_manager import DBManager


class BayesianEstimate:
    def __init__(self):
        self.basic_graph = None
        self.time_frame = 6000000
        self.multi_traffic_graph = []
        self.roads_dict = {}
        self.intersection_dict = {}
        self.db_handler = DBManager()
        self.get_basic_traffic_graph()

    def get_basic_traffic_graph(self):
        road_data = [get_road_data(road_file) for road_file in
                     Path(ROAD_DATA_PATH).iterdir()]  # save road data iterator
        intersection_data = [get_intersection_data(intersection_file) for intersection_file in
                             Path(INTERSEC_DATA_PATH).iterdir()]  # save intersection data iterator
        graph_data = [get_graph_data(graph_file) for graph_file in
                      Path(GRAPH_DATA).iterdir()]

        traffic_graph = nx.DiGraph()

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
            print(road_one)
            link_id = road_one["link_id"]
            del road_one["link_id"]
            traffic_graph.add_node(link_id, road_attr=tuple([value for value in road_one.values()]))

        for graph_iter in graph_data:
            for edge in graph_iter:
                traffic_graph.add_edge(edge[0], edge[1], intersec_attr=tuple([
                    value for value in self.intersection_dict[edge[2]].values()]))

        self.basic_graph = traffic_graph

    def generate_multi_traffic_graph(self):
        start_timestamp = None
        end_timestamp = None
        current_num = 0
        step_length = 100
        finish_one_flag = False
        first_exec = True
        matched_roads_num = 0
        self.multi_traffic_graph.clear()
        copy_traffic_graph = copy.deepcopy(self.basic_graph)
        data_total = self.db_handler.exec_sql(f"SELECT count(*) FROM history_road_data")[0][0]

        while current_num < data_total:
            history_data = self.db_handler.exec_sql(
                f"SELECT * FROM history_road_data limit {current_num},{step_length}")
            current_num += step_length

            if first_exec:
                start_timestamp = min(map(int, json.loads(history_data[0][1]).keys()))
                end_timestamp = start_timestamp + self.time_frame
                first_exec = False

            if finish_one_flag:
                self.multi_traffic_graph.append(copy_traffic_graph)
                copy_traffic_graph = copy.deepcopy(self.basic_graph)
                start_timestamp = min(map(int, json.loads(history_data[0][1]).keys()))
                end_timestamp = start_timestamp + self.time_frame
                finish_one_flag = False

            for road_one in history_data:
                road_id = int(road_one[0])
                history_speed_dict = json.loads(road_one[1])
                for timestamp in history_speed_dict.keys():
                    if start_timestamp <= int(timestamp) <= end_timestamp:
                        copy_traffic_graph.nodes[road_id]["history_speed"] = history_speed_dict[timestamp]
                        matched_roads_num += 1
                        break

            # for road_one in history_data:
            #     fro = int(road_one[3])
            #     to = int(road_one[4])
            #     history_speed_dict = json.loads(road_one[1])
            #
            #     for timestamp in history_speed_dict.keys():
            #         if start_timestamp <= int(timestamp) <= end_timestamp:
            #             copy_traffic_graph.edges[fro, to]["history_speed"] = history_speed_dict[timestamp]
            #             matched_roads_num += 1
            #             break

            # 匹配完成标志，如果考虑计算复杂度，可设置差值
            if matched_roads_num >= len(copy_traffic_graph.nodes) // 4:
                print(f"The match is successful <{matched_roads_num}>")
                finish_one_flag = True
                matched_roads_num = 0

        # for graph in self.multi_traffic_graph:
        #     print(graph)

        # print(self.multi_traffic_graph)


if __name__ == '__main__':
    BayesianEstimate().generate_multi_traffic_graph()


