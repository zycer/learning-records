import random

import numpy as np
# from get_data import GPSData
from kd_tree import KNN
from queue import PriorityQueue
import math
import os
from random import uniform, randint
from collections import OrderedDict


class Vertex:
    def __init__(self, idx, latitude=None, longitude=None):
        self.idx = idx
        self.come = 0
        self.out = 0
        self.latitude = latitude
        self.longitude = longitude


class RoadSegment:
    def __init__(self, idx, fro, to, name, speed_limit, road_nodes, mileage, average_speed):
        self.idx = idx
        self.fro = fro
        self.to = to
        self.name = name
        self.mileage = mileage
        self.average_speed = average_speed
        self.speed_limit = speed_limit
        self.road_nodes = road_nodes


class RoadNetworkGraph:
    def __init__(self):
        self.matrix = []
        self.vertex = {}
        self.road_segment = {}
        self.adjacency_table = {}
        self.inverse_adjacency_table = {}
        self.adjacency_matrix = []
        self.file_path = "data/road_graph"
        self.data_files = os.listdir(self.file_path)

    def save_road_network_data(self):
        """
        保存路网中的节点与边到成员变量中
        :return:
        """
        self.vertex.clear()
        self.road_segment.clear()

        for segment in self.matrix:
            idx, fro, to, name, mileage, average_speed = segment
            if fro not in self.vertex:
                # 测试使用，随机节点经纬度
                latitude = uniform(22, 23)
                longitude = uniform(113, 114)
                self.vertex[fro] = Vertex(fro, latitude, longitude)
            if to not in self.vertex:
                # 测试使用，随机节点经纬度
                latitude = uniform(22, 23)
                longitude = uniform(113, 114)
                self.vertex[to] = Vertex(to, latitude, longitude)

            self.vertex[fro].out += 1
            self.vertex[to].out += 1

            speed_limit = randint(40, 100)
            self.road_segment[idx] = RoadSegment(idx, fro, to, name, speed_limit, None, mileage, average_speed)

    def create_graph_adjacency_table(self):
        """
        邻接表存储图结构
        {from_vertex_id: {to_vertex_id: segment_object, to_vertex_id: segment_object}
        from_vertex_id: {to_vertex_id: segment_object, to_vertex_id: segment_object}...
        }
        :return:
        """
        self.save_road_network_data()
        for segment in self.road_segment.values():
            from_vertex = segment.fro
            to_vertex = segment.to
            if from_vertex not in self.adjacency_table.keys():
                self.adjacency_table[from_vertex] = dict()

            if to_vertex not in self.inverse_adjacency_table.keys():
                self.inverse_adjacency_table[to_vertex] = dict()

            self.adjacency_table[from_vertex].update(
                {to_vertex: segment}
            )

            self.inverse_adjacency_table[to_vertex].update(
                {from_vertex: segment}
            )

    def create_graph_adjacency_matrix(self):
        """
        邻接矩阵存储图结构
        :return:
        """
        self.save_road_network_data()
        for i in range(len(self.vertex)):
            self.adjacency_matrix.append([])
            for j in range(len(self.vertex)):
                self.adjacency_matrix[i].append(-1)

        for segment in self.road_segment.values():
            self.adjacency_matrix[segment.fro - 1][segment.to - 1] = segment

    def load_road_data(self):
        """
        加载文件中的路网信息，并建立路网图
        """

        for file in self.data_files:
            try:
                with open(os.path.join(self.file_path, file), encoding="utf-8") as f:
                    road_data = f.readlines()
            except Exception as e:
                print(e)
                road_data = None

            assert road_data
            road_data = road_data[1:]

            for segment in road_data:
                segment_attr = segment.split(",")
                segment_id = int(segment_attr[0].strip())
                from_vertex = int(segment_attr[2].strip())
                to_vertex = int(segment_attr[3].strip())
                segment_name = segment_attr[11].strip()
                mileage = float(segment_attr[-2].strip())
                average_speed = float(segment_attr[-1].strip()) if segment_attr[-1].strip() != '' else -1
                self.matrix.append([segment_id, from_vertex, to_vertex, segment_name, mileage, average_speed])

        # self.create_graph_adjacency_matrix()
        self.create_graph_adjacency_table()

    def neighbors(self, vertex_id):
        """
        获取节点邻居
        :param vertex_id: 节点id
        :return: 节点所有邻居
        """
        return self.adjacency_table[vertex_id].keys()

    def heuristic(self, from_vertex_id, to_vertex_id):
        """
        使用欧氏距离计算两点之间的启发式预估代价
        :param from_vertex_id: 起点节点id
        :param to_vertex_id: 终点节点id
        :return: 两点的预估代价
        """
        return math.hypot(self.vertex[from_vertex_id].latitude - self.vertex[to_vertex_id].latitude,
                          self.vertex[from_vertex_id].longitude - self.vertex[to_vertex_id].longitude)

    def show_adjacency_table(self):
        for key, value in self.adjacency_table.items():
            print(key, value)

    def show_graph_data(self):
        for key, vertex in self.vertex.items():
            print(f"{key}: {vertex.latitude}, {vertex.longitude}")
        print()
        for key, segment in self.road_segment.items():
            print(f"{key}: {segment.name}---{segment.mileage}")

        # for key, value in self.adjacency_table.items():
        #     print(f"{key}: {value}")

    def shortest_path(self, start_id, goal_id):
        """
        启发式查找两点之间的最短路径
        :param start_id:
        :param goal_id:
        :return:
        """

        class TempPriority:
            def __init__(self, vertex_id, cost):
                self.vertex_id = vertex_id
                self.priority = cost

            def __lt__(self, other):
                return self.priority < other.priority

        frontier = PriorityQueue()
        frontier.put(TempPriority(start_id, 0))

        came_from = OrderedDict()
        cost_so_far = dict()
        came_from[start_id] = None
        cost_so_far[start_id] = 0

        while not frontier.empty():
            current = frontier.get()

            if current.vertex_id == goal_id:
                break

            try:
                neighbors = self.neighbors(current.vertex_id)
            except KeyError:
                return []

            for next_vertex_id in neighbors:
                if next_vertex_id == goal_id:
                    new_cost = 0
                else:
                    new_cost = cost_so_far[current.vertex_id] + \
                               float(self.adjacency_table[current.vertex_id][next_vertex_id].mileage)
                if next_vertex_id not in cost_so_far or new_cost < cost_so_far[next_vertex_id]:
                    cost_so_far[next_vertex_id] = new_cost
                    priority = new_cost + self.heuristic(current.vertex_id, next_vertex_id)
                    frontier.put(TempPriority(next_vertex_id, priority))
                    came_from[next_vertex_id] = current.vertex_id
                if next_vertex_id == goal_id:
                    break

        return list(came_from.keys())

    def shortest_path_length(self, start_id, goal_id):
        """
        最短路径长度
        :return:
        """
        return len(self.shortest_path(start_id, goal_id))

    def average_speed_spl(self, start_id, goal_id):
        """
        最短路径上车辆行驶的平均速度
        :return:
        """
        # 最初没有数据，假设车辆平均行驶速度等于限速的平均值
        average_speed_list = []
        shortest_path = self.shortest_path(start_id, goal_id)
        for i in range(len(shortest_path) - 1):
            segment = self.adjacency_table[shortest_path[i]][shortest_path[i + 1]]
            if segment.average_speed != -1:
                average_speed_list.append(segment.average_speed)
            else:
                average_speed_list.append(segment.speed_limit)

        return sum(average_speed_list) / len(average_speed_list)

    def weighted_speed_limit_spl(self, start_id, goal_id):
        """
        最短路径上车辆的加权速度限制
        :return:
        """
        speed_limit = []
        shortest_path = self.shortest_path(start_id, goal_id)
        for i in range(len(shortest_path) - 1):
            segment = self.adjacency_table[shortest_path[i]][shortest_path[i + 1]]
            speed_limit.append(segment.speed_limit)

        return sum(speed_limit) / len(speed_limit)

    def k_nearest_neighbors(self, trajectory: list):
        """
        返回k个距离gps轨迹点最近的路段(KNN算法)
        :param trajectory: GPS轨迹点 [(x1,y1), (x2,y2)...]
        :return:
        """
        # 轨迹点附近的节点 [[1, 2], [3, 4]...]
        vertex_trajectory_range = []
        for point in trajectory:
            x, y = point
            scope_vertex = []
            for vertex_id, vertex in self.vertex.items():
                if math.hypot(x - vertex.longitude, y - vertex.latitude) < 0.01:
                    scope_vertex.append(vertex_id)
            vertex_trajectory_range.append(scope_vertex)

        # 根据轨迹带你附近的节点，查询其出度与入度，并将关联的路段保存[{1:segment_obj, 2: segment_obj}, {...,...},...]
        segment_trajectory_range = []

        for vertexes in vertex_trajectory_range:
            scope_segment = {}
            for vertex_id in vertexes:
                for segment in self.adjacency_table[vertex_id].values():
                    if segment.idx not in scope_segment.keys():
                        scope_segment[segment.idx] = segment

                for segment in self.inverse_adjacency_table[vertex_id].values():
                    if segment.idx not in scope_segment.keys():
                        scope_segment[segment.idx] = segment

            segment_trajectory_range.append(scope_segment)

        KNN(trajectory, segment_trajectory_range).matched_segments()

    def road_gps_point_located(self, gps_point):
        """
        GPS点(候选点)所在的道路
        :return: 返回路段的id
        """
        pass

    def road_speed_limit(self, gps_point):
        """
        通过路段id获取该路段限速
        :param gps_point:
        :return:
        """
        return self.road_segment[self.road_gps_point_located(gps_point)].speed_limit


class AIVMM:
    def __init__(self, graph: RoadNetworkGraph, mu, sigma):
        self.road_graph = graph
        self.mu = mu
        self.sigma = sigma

    # 位置和道路分析
    @classmethod
    def euclid_distance(cls, point_a, point_b):
        """
        欧几里得距离
        :param point_a: 点A
        :param point_b: 点B
        :return: 欧几里得距离
        """
        x1, y1 = point_a
        x2, y2 = point_b
        return math.hypot(x1 - x2, y1 - y2)

    def gps_observation_probability(self, candidate_point_i, candidate_point_j):
        """
        GPS点的观测概率
        param: candidate_point_i: 候选点i
        param: candidate_point_j: 候选点j
        """
        euclid_distance_ij = self.euclid_distance(candidate_point_i, candidate_point_j)
        return np.exp(-(euclid_distance_ij - self.mu) ** 2 / 2 * (self.sigma ** 2))

    def get_shortest_path_length(self, start_id, goal_id):
        """
        获取最短路径长度
        :return:
        """
        return self.road_graph.shortest_path_length(start_id, goal_id)

    def get_average_speed_spl(self, start_id, goal_id):
        """
        获取最短路径上车辆行驶的平均速度
        :return:
        """
        return self.road_graph.average_speed_spl(start_id, goal_id)

    def get_weighted_speed_limit_spl(self, start_id, goal_id):
        """
        获取最短路径上车辆的加权速度限制
        :return:
        """
        return self.road_graph.weighted_speed_limit_spl(start_id, goal_id)

    def get_road_speed_limit(self, gps_point):
        """
        获取GPS点所在路段的限速
        :param gps_point:
        :return:
        """
        return self.road_graph.road_speed_limit(gps_point)

    def excess_probability(self, candidate_point_i, candidate_point_j):
        """
        过滤概率函数
        param: candidate_point_i: 候选点i
        param: candidate_point_j: 候选点j
        """
        euclid_distance_ij = self.euclid_distance(candidate_point_i, candidate_point_j)
        return euclid_distance_ij / self.get_shortest_path_length(candidate_point_i, candidate_point_j)

    def spatial_analysis(self, candidate_point_i, candidate_point_j):
        """
        空间分析函数
        param: gop: GPS点的观测概率
        param: ep: 过度概率
        """
        gop = self.gps_observation_probability(candidate_point_i, candidate_point_j)
        ep = self.excess_probability(candidate_point_i, candidate_point_j)
        return gop * ep

    def time_analysis(self, candidate_point_i, candidate_point_j):
        """
        时间分析函数
        param: candidate_point_i: 候选点i
        param: candidate_point_j: 沿着候选点j
        """
        ass = self.get_average_speed_spl(candidate_point_i, candidate_point_j)
        wsls = self.get_weighted_speed_limit_spl(candidate_point_i, candidate_point_j)
        return ass / (abs(ass - wsls) + ass)

    def road_level_factor(self, candidate_point_i, candidate_point_j):
        """
        道路水平系数RLF
        param: vs: 候选点c_i-1所在道路限速
        param: vd: 候选点c_i所在道路限速
        """
        segment_i_speed_limits = self.get_road_speed_limit(candidate_point_i)
        segment_j_speed_limits = self.get_road_speed_limit(candidate_point_j)
        return segment_i_speed_limits / ((segment_j_speed_limits - segment_i_speed_limits) + segment_i_speed_limits)

    def path_weight(self, candidate_point_i, candidate_point_j):
        """
        param: fs: 空间分析函数
        param: ft: 时间分析函数
        param: rlf: 道路水平因子
        """
        sa = self.spatial_analysis(candidate_point_i, candidate_point_j)
        ta = self.time_analysis(candidate_point_i, candidate_point_j)
        rlf = self.road_level_factor(candidate_point_i, candidate_point_j)
        return sa * ta * rlf

    # 相互影响分析


def test_knn():
    res = []
    points = []
    for i in range(2, 4):
        points.append([random.uniform(100, 200), random.uniform(20, 60)])
        temp = {}
        for j in range(4, 7):
            idx = j if i == 2 else j * i
            road_nodes = []
            for k in range(20):
                road_nodes.append([random.uniform(100, 200), random.uniform(20, 60)])
            segment = RoadSegment(idx, 0, 0, "xxx", 60, road_nodes, 15, 55)
            temp[idx] = segment

        res.append(temp)
    print("数据生成...")
    KNN(points, res, 10).matched_segments()


if __name__ == "__main__":
    test_knn()
    # road_graph = RoadNetworkGraph()
    # road_graph.load_road_data()
    # road_graph.show_adjacency_table()
    # print(road_graph.shortest_path(1, 5))
    # print(road_graph.average_speed_spl(1, 5))
    # print(road_graph.weighted_speed_limit_spl(1, 5))
