import json
from pathlib import Path


import networkx as nx
import pandas as pd
from constants import ROAD_DATA_PATH, INTERSEC_DATA_PATH, ROAD_ATTR, INTERSEC_ATTR
from utils import get_road_data, get_intersection_data
from db_manager import DBManager


class BayesianEstimate:
    def __init__(self):
        self.basic_graph = None
        self.time_frame = 600
        self.db_handler = DBManager()

    def get_basic_traffic_graph(self):
        road_data = [get_road_data(road_file) for road_file in Path(ROAD_DATA_PATH).iterdir()]  # save road data iterator
        intersection_data = [get_intersection_data(intersection_file) for intersection_file in
                             Path(INTERSEC_DATA_PATH).iterdir()]    # save intersection data iterator
        traffic_graph = nx.DiGraph()

        for road_iter in road_data:
            for road_one_info in road_iter:
                road_attr_dict = dict(zip(ROAD_ATTR, road_one_info))
                traffic_graph.add_edge(road_attr_dict["from_node_id"], road_attr_dict["to_node_id"], **road_attr_dict)

        for intersec_iter in intersection_data:
            for intersec_one_info in intersec_iter:
                intersec_attr_dict = dict(zip(INTERSEC_ATTR, intersec_one_info))
                traffic_graph.add_node(intersec_attr_dict["node_id"], **intersec_attr_dict)

        self.basic_graph = traffic_graph

    def generate_multi_traffic_graph(self):
        history_data_dict = {}
        multi_traffic_graph = []
        start_timestamp = None
        current_num = 0
        step_length = 100
        finish_one_flag = False
        self.get_basic_traffic_graph()
        print(self.basic_graph.adj)

        data_total = self.db_handler.exec_sql(f"SELECT count(*) FROM history_road_data")[0][0]

        while current_num < data_total:
            next_num = current_num + step_length if current_num + step_length < data_total else data_total
            history_data = self.db_handler.exec_sql(f"SELECT * FROM history_road_data limit {current_num},{next_num}")
            current_num = next_num

            for road_one in history_data:
                start_timestamp = min(map(int, json.loads(road_one[1]).keys())) if \
                    finish_one_flag and road_one[1] else start_timestamp

                road_one[0]





        # for data in history_data:
        #

        # for data in history_data:
        #     history_data_dict[data[0]] = (json.loads(data[1]), data[2])
        # print(history_data_dict)


BayesianEstimate().generate_multi_traffic_graph()
