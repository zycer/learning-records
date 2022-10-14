import copy
import json
from collections import OrderedDict

from map_matching.utils.road_network import BaseRoadNetwork
from map_matching.utils.db_manager import DBManager
import networkx as nx


class MultiRoadNetwork(BaseRoadNetwork):
    def __init__(self, usage):
        self.db_manager = DBManager()
        # self.multi_road_network = []
        self.max_length = 0
        self.group_road_data = self.time_group_data()
        super().__init__(usage)

    def time_group_data(self):
        sql = "SELECT * FROM history_road_data"
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
        multi_flag = int(self.db_manager.exec_sql("SELECT multi_num FROM multi_flag")[0][0])

        print(f"从{multi_flag}条记录开始生成行驶速度路网图...")

        for i in range(multi_flag, self.max_length):
            one_network = copy.deepcopy(self.road_graph)
            effective_road_num = 0
            for road_id, one_road_data in self.group_road_data.items():
                if one_road_data:
                    effective_road_num += 1
                    one_network.nodes[road_id]["average_speed"] = one_road_data.popitem(last=False)[1]

            percentage = round(effective_road_num / total_road_num, 2)
            nx.write_graphml(one_network, f"data/multi_graph/road_graph_{i}_{percentage}.graphml")
            print(f"已持久化路网图：data/multi_graph/road_graph_{i}_{percentage}.graphml")

            self.db_manager.exec_sql("UPDATE multi_flag SET multi_num=multi_num+1 WHERE id=0")


if __name__ == '__main__':
    multi_network = MultiRoadNetwork("gcn")
    multi_network.time_group_data()

    multi_network.generate_multi_road_network()
