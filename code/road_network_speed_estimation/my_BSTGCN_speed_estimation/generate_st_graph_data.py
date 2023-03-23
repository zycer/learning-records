import pickle
import json
import sys
import math
import random
import threading

import tqdm
from collections import OrderedDict
from datetime import datetime

from map_matching.utils.road_network import BaseRoadNetwork
from map_matching.utils.db_manager import DBManager


class MultiRoadNetwork(BaseRoadNetwork):
    def __init__(self, usage):
        self.db_manager = DBManager()
        self.min_timestamp = sys.maxsize
        self.time_step = 1800
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
        print("\ngenerate_st_road_graph...")
        time_road_data = OrderedDict()
        for road_id, one_values in tqdm.tqdm(self.all_road_data.items()):
            for timestamp, speed in one_values.items():
                time_step_num = math.ceil((timestamp - self.min_timestamp) / self.time_step)
                time_window = self.min_timestamp + time_step_num * self.time_step
                if time_window not in time_road_data.keys():
                    time_road_data[time_window] = {}
                if road_id not in time_road_data[time_window].keys():
                    time_road_data[time_window][road_id] = []
                time_road_data[time_window][road_id].append(speed)

        return time_road_data


def save_graph_data(_data):
    __index, __timestamp_list, __data = _data
    dump_data = {}
    for temp_data in zip(__timestamp_list, __data):
        dump_data[temp_data[0]] = temp_data[1]
    with open(f"{data_path}/{__index}.pickle", "wb") as f:
        pickle.dump(dump_data, f)


if __name__ == '__main__':
    threads = []
    multi_network = MultiRoadNetwork("gcn")
    multi_network.init_graph()
    time_road_data = multi_network.generate_st_road_graph()
    data_path = "../road_network_speed_estimation/my_BSTGCN_speed_estimation/data"

    print("\nComplete data...")
    timestamp_list = []
    num = 0
    for index, __values in tqdm.tqdm(enumerate(time_road_data.items()), total=len(time_road_data)):
        timestamp, values = __values
        for road_id, speeds in values.items():
            average_speed = sum(speeds) / len(speeds)
            time_road_data[timestamp][road_id] = average_speed

        no_data_road_ids = list(set(multi_network.road_graph.nodes).difference(set(values.keys())))
        for _road_id in no_data_road_ids:
            if _road_id in multi_network.all_road_data.keys():
                if multi_network.all_road_data[_road_id]:
                    flag = False
                    for _timestamp, _speed in multi_network.all_road_data[_road_id].items():
                        dt1 = datetime.fromtimestamp(timestamp)
                        dt2 = datetime.fromtimestamp(_timestamp)
                        minutes1 = dt1.hour * 60 + dt1.minute
                        minutes2 = dt2.hour * 60 + dt2.minute
                        if abs(minutes2 - minutes1) <= multi_network.time_step * 2:
                            time_road_data[timestamp][_road_id] = _speed + random.uniform(-5, 5)
                            flag = True
                            break
                    if not flag:
                        sum(multi_network.all_road_data[_road_id].values()) / len(
                            multi_network.all_road_data[_road_id].values())
                else:
                    time_road_data[timestamp][_road_id] = multi_network.road_graph.nodes[_road_id][
                                                              "free_speed"] + random.uniform(-5, 5)
            else:
                time_road_data[timestamp][_road_id] = multi_network.road_graph.nodes[_road_id][
                                                          "free_speed"] + random.uniform(-5, 5)
            if time_road_data[timestamp][_road_id] < 10:
                time_road_data[timestamp][_road_id] = multi_network.road_graph.nodes[_road_id][
                                                          "free_speed"] + random.uniform(-5, 5)
        timestamp_list.append(timestamp)
        if len(timestamp_list) == 100:
            t = threading.Thread(target=save_graph_data,
                                 args=((num, timestamp_list, [time_road_data[temp] for temp in timestamp_list]),))
            threads.append(t)
            t.start()
            timestamp_list.clear()
            num += 1

    if timestamp_list:
        t = threading.Thread(target=save_graph_data,
                             args=((num, timestamp_list, [time_road_data[temp] for temp in timestamp_list]),))
        threads.append(t)
        t.start()

    print("\nWait for the child process to end...")
    for t in tqdm.tqdm(threads):
        t.join()
