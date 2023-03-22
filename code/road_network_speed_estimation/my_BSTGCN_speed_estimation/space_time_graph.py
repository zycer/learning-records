import copy
import json
import sys
import math
import random
import tqdm
from collections import OrderedDict
from datetime import datetime

from map_matching.utils.road_network import BaseRoadNetwork
from map_matching.utils.db_manager import DBManager
import networkx as nx


class MultiRoadNetwork(BaseRoadNetwork):
    def __init__(self, usage):
        self.db_manager = DBManager()
        self.min_timestamp = sys.maxsize
        self.time_step = 408800
        self.all_road_data = self.get_all_road_data()
        super().__init__(usage)

    def get_all_road_data(self):
        print("get_all_road_data.....")
        sql = "SELECT * FROM history_road_data"
        query_data = self.db_manager.exec_sql(sql)
        road_data_dict = {}
        for one_data in tqdm.tqdm(query_data):
            in_one_road = OrderedDict()
            one_speed_data = {}
            for _dict in one_data[1].split(";"):
                one_speed_data.update(json.loads(_dict))

            for key in sorted([int(key) for key in one_speed_data.keys()]):
                in_one_road[key] = one_speed_data[str(key)]
                if key < self.min_timestamp:
                    self.min_timestamp = key
            road_data_dict[int(one_data[0])] = in_one_road

        return road_data_dict

    def generate_st_road_graph(self):
        print("generate_st_road_graph...")
        self.init_graph()
        time_road_data = OrderedDict()
        print("generate timestamp graph...")
        for road_id, one_values in tqdm.tqdm(self.all_road_data.items()):
            for timestamp, speed in one_values.items():
                time_step_num = math.ceil((timestamp - self.min_timestamp) / self.time_step)
                time_window = self.min_timestamp + time_step_num * self.time_step
                if time_window not in time_road_data.keys():
                    time_road_data[time_window] = {}
                if road_id not in time_road_data[time_window].keys():
                    time_road_data[time_window][road_id] = []
                time_road_data[time_window][road_id].append(speed)

        print("补全数据。。。")
        for timestamp, values in tqdm.tqdm(time_road_data.items()):
            for road_id, speeds in values.items():
                average_speed = sum(speeds) / len(speeds)
                time_road_data[timestamp][road_id] = average_speed

            no_data_road_ids = list(set(self.road_graph.nodes).difference(set(values.keys())))
            for _road_id in no_data_road_ids:
                if _road_id in self.all_road_data.keys():
                    if self.all_road_data[_road_id]:
                        flag = False
                        for _timestamp, _speed in self.all_road_data[_road_id].items():
                            dt1 = datetime.fromtimestamp(timestamp)
                            dt2 = datetime.fromtimestamp(_timestamp)
                            minutes1 = dt1.hour * 60 + dt1.minute
                            minutes2 = dt2.hour * 60 + dt2.minute
                            if abs(minutes2 - minutes1) <= self.time_step * 2:
                                time_road_data[timestamp][_road_id] = _speed + random.uniform(-5, 5)
                                flag = True
                                break
                        if not flag:
                            sum(self.all_road_data[_road_id].values()) / len(self.all_road_data[_road_id].values())
                    else:
                        time_road_data[timestamp][_road_id] = self.road_graph.nodes[_road_id][
                                                                  "free_speed"] + random.uniform(-5, 5)
                else:
                    time_road_data[timestamp][_road_id] = self.road_graph.nodes[_road_id][
                                                              "free_speed"] + random.uniform(-5, 5)

        # 只留长度、车道数、限速
        for road_id in self.road_graph.nodes:
            del self.road_graph.nodes[road_id]["from_node_id"]
            del self.road_graph.nodes[road_id]["to_node_id"]
            del self.road_graph.nodes[road_id]["average_speed"]
            self.road_graph.nodes[road_id]["time_data_speed"] = []

        # 写入图...
        print("写入图...")
        for timestamp, values in tqdm.tqdm(time_road_data.items()):
            for road_id, speed in values.items():
                self.road_graph.nodes[road_id]["time_data_speed"].append((timestamp, speed))

        for road_id in self.road_graph.nodes:
            time_data_json = json.dumps(self.road_graph.nodes[road_id]["time_data_speed"])
            self.road_graph.nodes[road_id]["time_data_speed"] = time_data_json

        nx.write_graphml(self.road_graph, f"data/st_road_graph.graphml")


if __name__ == '__main__':
    multi_network = MultiRoadNetwork("gcn")
    multi_network.generate_st_road_graph()
