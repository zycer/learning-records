import copy
import json
from collections import OrderedDict

from road_network import BaseRoadNetwork
from db_manager import DBManager


class MultiRoadNetwork(BaseRoadNetwork):
    def __init__(self, usage):
        self.db_manager = DBManager()
        self.multi_road_network = []
        self.max_length = 0
        self.group_road_data = self.time_group_data()
        super().__init__(usage)

    def time_group_data(self):
        sql = "SELECT * FROM history_road_data limit 0, 10000"
        query_data = self.db_manager.exec_sql(sql)
        road_data_dict = {}
        for one_data in query_data:
            in_one_road = OrderedDict()
            one_speed_data = {}
            for _dict in one_data[1].split(";"):
                one_speed_data.update(json.loads(_dict))

            if len(one_speed_data) > self.max_length:
                self.max_length = len(one_speed_data)

            for key in sorted(one_speed_data.keys()):
                in_one_road[key] = one_speed_data[key]
            road_data_dict[int(one_data[0])] = in_one_road
        return road_data_dict

    def generate_multi_road_network(self):
        self.init_graph()
        total_road_num = int(self.db_manager.exec_sql("SELECT COUNT(*) FROM history_road_data")[0][0])
        for i in range(self.max_length):

            one_network = copy.deepcopy(self.road_graph)
            effective_road_num = 0
            for road_id, one_road_data in self.group_road_data.items():
                if one_road_data:
                    effective_road_num += 1
                    one_network.nodes[road_id]["road_attr"][-1] = one_road_data.popitem(last=False)

            if effective_road_num / total_road_num >= 0.008:
                self.multi_road_network.append(one_network)

            print(len(self.group_road_data[11212]), self.group_road_data[11212])

            if i == 5:
                break

        for i in self.multi_road_network:
            print(i.nodes[11212])


if __name__ == '__main__':
    multi_network = MultiRoadNetwork("gcn")
    multi_network.time_group_data()

    multi_network.generate_multi_road_network()
