import json
import math
import time
import os
import numpy as np
import pandas as pd
import redis
import requests

from kd_tree2 import KNN
from queue import PriorityQueue
from collections import OrderedDict
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from db_manager import DBManager
from constants import ROAD_MAX_SPEED as RMS
from utils import NpEncoder

URL = "http://127.0.0.1:8000/"
road_temp_data = json.loads(requests.get(URL + "get_all_roads_info").text)
ALL_ROAD_INFO = {}
for road_id, road_info in road_temp_data.items():
    ALL_ROAD_INFO[int(road_id)] = road_info


class CandidateGraph:
    class Vertex:
        def __init__(self, idx):
            self.idx = idx
            self.observation_probability = None

    class Edge:
        def __init__(self, idx, fro, to):
            self.fro = fro
            self.to = to
            self.idx = idx
            # self.weight = None
            self.vote = 0

    def __init__(self, check_legitimate_path_func, observation_probability_func, path_weight_func):
        self.vertex = {}
        self.edge = {}
        self.adjacency_table = {}
        self.check_legitimate_path_func = check_legitimate_path_func
        self.observation_probability_func = observation_probability_func
        self.path_weight_func = path_weight_func

    def generate_candidate_graph(self, trajectory, candidate_roads, candidate_points):
        for i in range(len(candidate_points) - 1):
            for j, point_j in enumerate(candidate_points[i]):
                self.adjacency_table[f"{i}&{j}"] = {}
                for k, point_k in enumerate(candidate_points[i + 1]):
                    if f"{i}&{j}" not in self.vertex:
                        self.vertex[f"{i}&{j}"] = self.Vertex(f"{i}&{j}")
                        observation_probability = self.observation_probability_func(point_j, trajectory[i])
                        self.vertex[f"{i}&{j}"].observation_probability = observation_probability

                    if f"{i + 1}&{k}" not in self.vertex:
                        self.vertex[f"{i + 1}&{k}"] = self.Vertex(f"{i + 1}&{k}")
                        observation_probability = self.observation_probability_func(point_k, trajectory[i + 1])
                        self.vertex[f"{i + 1}&{k}"].observation_probability = observation_probability
                    if self.check_legitimate_path_func(ALL_ROAD_INFO[candidate_roads[i][j]][1],
                                                       ALL_ROAD_INFO[candidate_roads[i + 1][k]][0]):
                        edge_id = f"{i}&{j}|{i + 1}&{k}"
                        edge = self.Edge(edge_id, f"{i}&{j}", f"{i + 1}&{k}")
                        # edge.weight = self.path_weight_func(trajectory[i], trajectory[i + 1],
                        #                                     candidate_points[i + 1][j], candidate_roads[i][j],
                        #                                     candidate_roads[i + 1][k])
                        self.edge[edge_id] = edge
                        self.adjacency_table[f"{i}&{j}"][f"{i + 1}&{k}"] = edge_id

    def show_data(self):
        print("-" * 140)
        print("候选图：")
        table = PrettyTable(["边", "起点", "终点", "权重"])
        for key, value in self.edge.items():
            table.add_row([value.idx, value.fro, value.to, value.weight])
        print("%s\n" % table)
        table = PrettyTable(["顶点", "观测概率"])
        for key, value in self.vertex.items():
            table.add_row([key, value.observation_probability])
        print("%s\n" % table)
        print("-" * 140)


class RoadNetworkGraph:
    class Vertex:
        def __init__(self, idx, name, longitude=None, latitude=None):
            self.name = name
            self.idx = idx
            self.come = 0
            self.out = 0
            self.latitude = latitude
            self.longitude = longitude

    class RoadSegment:
        def __init__(self, idx, fro, to, name, mileage, speed_limit, average_speed, road_nodes):
            self.idx = idx
            self.fro = fro
            self.to = to
            self.name = name
            self.mileage = mileage
            self.average_speed = average_speed
            self.speed_limit = speed_limit
            self.road_nodes = road_nodes

    def __init__(self):
        self.matrix = []
        self.vertex = {}
        self.road_segment = {}
        self.adjacency_table = {}
        self.inverse_adjacency_table = {}
        self.adjacency_matrix = []
        self.road_path = "data/road_data"
        self.vertex_path = "data/vertex_data"
        self.road_data_files = os.listdir(self.road_path)
        self.vertex_data_files = os.listdir(self.vertex_path)
        self.db_handler = DBManager()

    def save_road_network_data(self):
        """
        保存路网中的节点与边到成员变量中
        :return:
        """
        self.road_segment.clear()

        for road in self.matrix:
            idx, fro, to, name, mileage, speed_limit, average_speed, road_nodes = road
            assert fro in self.vertex
            assert to in self.vertex

            self.vertex[fro].out += 1
            self.vertex[to].out += 1

            self.road_segment[idx] = self.RoadSegment(idx, fro, to, name, mileage,
                                                      speed_limit, average_speed, road_nodes)

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
        res = self.db_handler.exec_sql(f"SELECT road_id, average_speed FROM history_road_data")
        known_roads = {}
        for data in res:
            known_roads[data[0]] = data[1]

        for vertex_file in self.vertex_data_files:
            vertex_data = pd.read_csv(os.path.join(self.vertex_path, vertex_file), encoding="utf-8", sep=",")
            longitudes = vertex_data["x_coord"].values
            latitudes = vertex_data["y_coord"].values
            vertex_names = vertex_data["name"].values
            vertex_ids = vertex_data["node_id"].values

            for vertex_one in zip(vertex_ids, vertex_names, longitudes, latitudes):
                vertex_id, vertex_name, longitude, latitude = vertex_one
                self.vertex[vertex_id] = self.Vertex(vertex_id, vertex_name, longitude, latitude)

        for road_file in self.road_data_files:
            road_data = pd.read_csv(os.path.join(self.road_path, road_file), encoding="utf-8", sep=",")
            road_names = road_data["name"].values
            road_ids = road_data["link_id"].values.tolist()
            from_vertexes = road_data["from_node_id"].values
            to_vertexes = road_data["to_node_id"].values
            geometries = np.array(list(map(lambda x: x[12:-1], road_data["geometry"].values)))
            mileages = road_data["length"].values
            speed_limits = np.array(
                list(map(lambda x: RMS.get(x, RMS["other"]), road_data["link_type_name"].values)))

            for road_one in zip(road_ids, from_vertexes, to_vertexes, road_names, mileages, speed_limits, geometries):
                road_id, fro, to, name, length, speed_limit, geometry = road_one

                if road_id in known_roads.keys():
                    average_speed = known_roads[road_id]
                else:
                    average_speed = 60
                road_nodes = []

                for points_str in geometry.split(","):
                    longitude, latitude = points_str.strip().split(" ")
                    road_nodes.append([float(longitude), float(latitude)])

                self.matrix.append([road_id, fro, to, name, length, speed_limit, average_speed, road_nodes])

        # self.create_graph_adjacency_matrix()
        self.create_graph_adjacency_table()
        # self.show_graph_data()

    def neighbors(self, vertex_id):
        """
        获取节点邻居
        :param vertex_id: 节点id
        :return: 节点所有邻居
        """
        try:
            return self.adjacency_table[vertex_id].keys()
        except KeyError:
            return []

    def heuristic(self, from_vertex_id, to_vertex_id):
        """
        使用欧氏距离计算两点之间的启发式预估代价
        :param from_vertex_id: 起点节点id
        :param to_vertex_id: 终点节点id
        :return: 两点的预估代价
        """
        # return math.hypot(self.vertex[from_vertex_id].latitude - self.vertex[to_vertex_id].latitude,
        #                   self.vertex[from_vertex_id].longitude - self.vertex[to_vertex_id].longitude)

        return geodesic((self.vertex[from_vertex_id].latitude, self.vertex[from_vertex_id].longitude),
                        (self.vertex[to_vertex_id].latitude, self.vertex[to_vertex_id].longitude)).km

    def show_adjacency_table(self):
        for key, value in self.adjacency_table.items():
            print(key, value)

    def show_graph_data(self, show_vertex=True):
        if show_vertex:
            print("路网数据：")
            table = PrettyTable(["路口", "经度", "纬度"])
            for key, vertex in self.vertex.items():
                table.add_row([key, vertex.longitude, vertex.latitude])
            print("%s\n" % table)
        if not show_vertex:
            print("更新后的路网数据：")
        table = PrettyTable(["路段", "名称", "起点", "终点", "限速", "平均车速"])
        for key, segment in self.road_segment.items():
            table.add_row([key, segment.name, segment.fro, segment.to, segment.speed_limit, segment.average_speed])
        print(table)
        print("-" * 140)

    def shortest_path(self, start, goal):
        """
        启发式查找两点之间的最短路径
        :param start:
        :param goal:
        :return:
        """
        if isinstance(start, RoadNetworkGraph.RoadSegment) and isinstance(goal, RoadNetworkGraph.RoadSegment):
            if start.idx == goal.idx:
                return [goal.fro, start.to]
            else:
                start_id = start.to
                goal_id = goal.fro
        else:
            start_id = start
            goal_id = goal

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
                while not frontier.empty():
                    del came_from[frontier.get().vertex_id]
                return list(came_from.keys())

            while not frontier.empty():
                vertex_id = frontier.get().vertex_id
                del came_from[vertex_id]
                del cost_so_far[vertex_id]

            for next_vertex_id in self.neighbors(current.vertex_id):
                new_cost = cost_so_far[current.vertex_id] + self.heuristic(current.vertex_id, next_vertex_id)
                if next_vertex_id not in cost_so_far or new_cost < cost_so_far[next_vertex_id]:
                    cost_so_far[next_vertex_id] = new_cost
                    priority = new_cost + self.heuristic(goal_id, next_vertex_id)
                    frontier.put(TempPriority(next_vertex_id, priority))
                    came_from[next_vertex_id] = current.vertex_id

        return []

    def shortest_path_length(self, start, goal):
        """
        最短路径长度
        :return:
        """
        if isinstance(start, RoadNetworkGraph.RoadSegment) and isinstance(goal, RoadNetworkGraph.RoadSegment):
            if start.idx == goal.idx:
                return 2
            else:
                return len(self.shortest_path(start.to, goal.fro))
        else:
            return len(self.shortest_path(start, goal))

    def average_speed_spl(self, start, goal):
        """
        最短路径上车辆行驶的平均速度
        :return:
        """
        # 最初没有数据，假设车辆平均行驶速度等于限速的平均值
        average_speed_list = []
        shortest_path = self.shortest_path(start, goal)
        for i in range(len(shortest_path) - 1):
            segment = self.adjacency_table[shortest_path[i]][shortest_path[i + 1]]
            if segment.average_speed != -1:
                average_speed_list.append(segment.average_speed)
            else:
                average_speed_list.append(segment.speed_limit)

        return sum(average_speed_list) / len(average_speed_list) if len(average_speed_list) else 0

    def weighted_speed_limit_spl(self, start, goal):
        """
        最短路径上车辆的加权速度限制
        :return:
        """
        speed_limit = []
        shortest_path = self.shortest_path(start, goal)
        for i in range(len(shortest_path) - 1):
            segment = self.adjacency_table[shortest_path[i]][shortest_path[i + 1]]
            speed_limit.append(segment.speed_limit)

        return sum(speed_limit) / len(speed_limit) if len(speed_limit) else 0

    def k_nearest_neighbors(self, trajectory: list, is_search=False):
        """
        返回k个距离gps轨迹点最近的路段(KNN算法)
        :param trajectory: GPS轨迹点 [(x1,y1), (x2,y2)...]
        :return:
        """
        return KNN(trajectory, self.road_segment).matched_knn(False)

    def road_speed_limit(self, road_id):
        """
        通过路段id获取该路段限速
        :param road_id:
        :return:
        """
        return self.road_segment[road_id].speed_limit


class AIVMM:
    def __init__(self, mu, sigma, beta, neighbor_num=3):
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.neighbor_num = neighbor_num
        self.candidate_graph_obj = None
        self.shortest_path_dict = {}
        # self.temp_shortest_path = []
        # self.temp_shortest_path_info = []

    # 位置和道路分析
    @classmethod
    def euclid_distance(cls, point_a, point_b):
        """
        欧几里得距离
        :param point_a: 点A
        :param point_b: 点B
        :return: 欧几里得距离
        """
        long_1, lat_1 = point_a
        long_2, lat_2 = point_b
        return geodesic((lat_1, long_1), (lat_2, long_2)).km

    def distance_weight(self, point_a, point_b):
        """
        两点之间的距离权重，公式为：exp(-(dist(pi,pj)^2 / beta^2))
        """
        euclid_distance = self.euclid_distance(point_a, point_b)
        return math.exp(-(euclid_distance ** 2 / self.beta ** 2))

    def gps_observation_probability(self, candidate_point, sample_point):
        """
        GPS点的观测概率
        param: candidate_point: 候选点
        param: sample_point: 采样点
        """
        euclid_distance = self.euclid_distance(candidate_point, sample_point)
        return np.exp(-((euclid_distance - self.mu) ** 2) / (2 * (self.sigma ** 2)))

    def get_shortest_path(self, start, goal):
        url = URL + f"get_shortest_path?start={start}&goal={goal}"
        result = json.loads(requests.get(url=url).text)
        if start in self.shortest_path_dict.keys():
            self.shortest_path_dict[start][goal] = result
        else:
            self.shortest_path_dict[start] = {goal: result}

    def get_shortest_path_length(self, start, goal):
        """
        获取最短路径长度
        :return:
        """
        return len(self.shortest_path_dict[start][goal][0])

    def get_average_speed_spl(self, start, goal):
        """
        最短路径上车辆行驶的平均速度
        :return:
        """
        # 最初没有数据，假设车辆平均行驶速度等于限速的平均值

        average_speed_list = list(map(lambda x: x[1] if x[1] != -1 else x[0], self.shortest_path_dict[start][goal][1]))
        return sum(average_speed_list) / len(average_speed_list) if len(average_speed_list) else 0

    def get_weighted_speed_limit_spl(self, start, goal):
        """
        获取最短路径上车辆的加权速度限制
        :return:
        """
        speed_limit = [i[0] for i in self.shortest_path_dict[start][goal][1]]
        return sum(speed_limit) / len(speed_limit) if len(speed_limit) else 0

    def get_road_speed_limit(self, road_id):
        """
        获取GPS点所在路段的限速
        :param gps_point:
        :return:
        """
        return ALL_ROAD_INFO[road_id][2]

    def excess_probability(self, sample_point_pre, sample_point_cur, pre_road_id, cur_road_id):
        """
        过滤概率函数
        :param sample_point_pre: 当前采样点前一个采样点
        :param sample_point_cur: 当前采样点
        :param pre_road_id: 前一个采样点的候选点所在道路的id
        :param cur_road_id: 当前采样点的候选点所在道路的id
        :return: 两个连续候选点之间的最短路径和直路径的相似性(过度概率)
        """
        euclid_distance = self.euclid_distance(sample_point_pre, sample_point_cur)
        shortest_path_length = self.get_shortest_path_length(ALL_ROAD_INFO[pre_road_id][1],
                                                             ALL_ROAD_INFO[cur_road_id][0])
        return euclid_distance / shortest_path_length if shortest_path_length else 0

    def spatial_analysis(self, sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_id, cur_road_id):
        """
        空间分析函数
        param: gop: GPS点的观测概率
        param: ep: 过度概率
        """
        gop = self.gps_observation_probability(candidate_point_cur, sample_point_cur)
        ep = self.excess_probability(sample_point_pre, sample_point_cur, pre_road_id, cur_road_id)
        return gop * ep

    def time_analysis(self, start, goal):
        """
        时间分析函数
        param: candidate_point_i: 候选点i
        param: candidate_point_j: 沿着候选点j
        """
        ass = self.get_average_speed_spl(start, goal)
        wsls = self.get_weighted_speed_limit_spl(start, goal)
        return ass / (abs(ass - wsls) + ass) if abs(ass - wsls) + ass else 0

    def road_level_factor(self, pre_road_id, cur_road_id):
        """
        道路水平系数RLF
        param: vs: 候选点c_i-1所在道路限速
        param: vd: 候选点c_i所在道路限速
        """

        segment_i_speed_limits = self.get_road_speed_limit(pre_road_id)
        segment_j_speed_limits = self.get_road_speed_limit(cur_road_id)
        return segment_i_speed_limits / ((segment_j_speed_limits - segment_i_speed_limits) + segment_i_speed_limits)

    def path_weight(self, sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_id, cur_road_id):
        self.get_shortest_path(ALL_ROAD_INFO[pre_road_id][1], ALL_ROAD_INFO[cur_road_id][0])
        sa = self.spatial_analysis(sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_id, cur_road_id)
        ta = self.time_analysis(ALL_ROAD_INFO[pre_road_id][1], ALL_ROAD_INFO[cur_road_id][0])
        rlf = self.road_level_factor(pre_road_id, cur_road_id)
        return sa * ta * rlf

    # 相互影响分析
    def static_score_matrix(self, trajectory, candidate_roads, candidate_points, is_show=False):
        """
        param: candidate_points: 轨迹的候选点
        静态评分矩阵
        """
        matrix_list = []
        for i in range(len(candidate_points) - 1):
            weight_list = []
            for j, point_j in enumerate(candidate_points[i]):
                for k, point_k in enumerate(candidate_points[i + 1]):
                    weight_list.append(self.path_weight(trajectory[i], trajectory[i + 1], point_k,
                                                        candidate_roads[i][j], candidate_roads[i + 1][k]))
            matrix = np.matrix(np.array(weight_list).reshape(
                len(candidate_points[i]), len(candidate_points[i + 1])), copy=True)
            matrix_list.append(matrix)

        score_matrix = np.matrix([])
        for matrix in matrix_list:
            little_mat1 = np.matrix(np.zeros([score_matrix.shape[0], matrix.shape[1]]))
            little_mat2 = np.matrix(np.zeros([matrix.shape[0], score_matrix.shape[1]]))
            little_mat1[:] = -np.inf
            little_mat2[:] = -np.inf
            score_matrix = np.bmat([[score_matrix, little_mat1], [little_mat2, matrix]])

        if is_show:
            print("静态评分矩阵: ")
            print(np.around(score_matrix[1:, :], 5))
            print("-" * 140)
        return matrix_list

    def distance_weight_matrix(self, trajectory, is_show=False):
        """
        轨迹点的距离权重矩阵
        """
        weight_matrix = []
        for i in range(len(trajectory)):
            omega_ij_list = []
            for j in range(len(trajectory)):
                if i != j:
                    omega_ij = self.distance_weight(trajectory[i], trajectory[j])
                    omega_ij_list.append(omega_ij)

            omega_i_matrix = np.diag(omega_ij_list)
            weight_matrix.append(omega_i_matrix)

        if is_show:
            print("距离评分矩阵：")
            for m in weight_matrix:
                print(m)
                print()
            print("-" * 140)
        return weight_matrix

    def weighted_scoring_matrix(self, trajectory, candidate_roads, candidate_points, is_show=False):
        """
        param: trajectory: GPS轨迹点
        param: candidate_points: GPS轨迹点对应的候选点
        加权得分矩阵
        """
        static_score_matrix = self.static_score_matrix(trajectory, candidate_roads, candidate_points)
        distance_weight_matrix = self.distance_weight_matrix(trajectory)
        phi_list = []
        phi_matrix_list = []

        for i in range(len(trajectory)):
            phi_i_list = []
            for j in range(len(trajectory) - 1):
                if 0 <= j <= i:
                    phi_ij = distance_weight_matrix[i][j - 1][j - 1] * static_score_matrix[j]
                else:
                    phi_ij = distance_weight_matrix[i][j][j] * static_score_matrix[j]
                phi_i_list.append(phi_ij)

            phi_list.append(phi_i_list)

        for phi_i in phi_list:
            weighted_score_matrix = np.matrix([])
            for phi_ij in phi_i:
                little_mat_1 = np.matrix(np.zeros([weighted_score_matrix.shape[0], phi_ij.shape[1]]))
                little_mat_2 = np.matrix(np.zeros([phi_ij.shape[0], weighted_score_matrix.shape[1]]))
                little_mat_1[:] = -np.inf
                little_mat_2[:] = -np.inf
                weighted_score_matrix = np.bmat([[weighted_score_matrix, little_mat_1], [little_mat_2, phi_ij]])
            phi_matrix_list.append(weighted_score_matrix[1:, :])

        if is_show:
            print("加权评分矩阵：")
            for i, matrix in enumerate(phi_matrix_list):
                print(f"采样点{i}:")
                print(np.around(matrix, 2))
                print()
            print("-" * 140)
        return distance_weight_matrix, phi_list

    def check_legitimate_path(self, road_a, road_b):
        """
        检查两点之间是否可达
        :param road_a: 点a
        :param road_b: 点b
        :return: 是否可达
        """
        return True if self.get_shortest_path_length(road_a, road_b) else False

    def find_local_optimal_path(self, omega_i, phi_i, candi_count, n, i, k):
        """
        获取局部最优路径
        :param omega_i: 距离权重矩阵
        :param phi_i: 权重评分矩阵
        :param candi_count: 各个采样点的候选点个数
        :param n: 采样点个数
        :param i:
        :param k:
        :return: 局部最优路径
        """
        f_ik = []
        pre_ik = []

        for ii in range(n):
            f_ik.append([])
            pre_ik.append([])
            for k in range(candi_count[i]):
                f_ik[ii].append(-np.inf)
                pre_ik[ii].append(-np.inf)

        for t in range(candi_count[0]):
            f_ik[0][t] = omega_i[0, 0] * self.candidate_graph_obj.vertex[f"{0}&{t}"].observation_probability

        for j in range(1, n):
            for s in range(candi_count[j]):
                f_ik[j][s] = max([f_ik[j - 1][t] + phi_i[j - 1][t, s] for t in range(candi_count[j - 1])])
                pre_ik[j][s] = np.argmax([f_ik[j - 1][t] + phi_i[j - 1][t, s] for t in range(candi_count[j - 1])])

        matched_path = []

        c = np.argmax([f_ik[n - 1][s] for s in range(candi_count[-1])])
        f_value_cik = max([f_ik[n - 1][s] for s in range(candi_count[-1])])

        for i in range(1, n).__reversed__():
            matched_path.append(c)
            c = pre_ik[i][c]

        matched_path.append(c)
        matched_path.reverse()
        return matched_path, f_value_cik

    def find_local_optimal_path_sequence(self, trajectory, candidate_roads, candidate_points):
        """
        获取局部最优路径序列（所有点的局部最优路径）
        :param trajectory: GPS轨迹点
        :param candidate_roads: 候选点所在的路段id
        :param candidate_points:    候选点
        :return: 列表，局部最优路径序列
        """
        local_optimal_path_sequence = []
        f_value_sequence = []

        candi_count = [len(points) for points in candidate_points]
        # haoshi
        distance_weight_matrix, phi_list = self.weighted_scoring_matrix(trajectory, candidate_roads, candidate_points)
        self.candidate_graph_obj = self.create_candidate_graph(trajectory, candidate_roads, candidate_points)
        for i, points in enumerate(candidate_points):
            phi_i = phi_list[i]
            lop_i = []
            f_value_i = []
            for k in range(len(points)):
                local_optimal_path, f_value = self.find_local_optimal_path(distance_weight_matrix[i], phi_i,
                                                                           candi_count, len(trajectory), i, k)
                lop_i.append(local_optimal_path)
                f_value_i.append(f_value)

            local_optimal_path_sequence.append(lop_i)
            f_value_sequence.append(f_value_i)

        return local_optimal_path_sequence, f_value_sequence

    def create_candidate_graph(self, trajectory, candidate_roads, candidate_points):
        candidate_graph_obj = CandidateGraph(self.check_legitimate_path, self.gps_observation_probability,
                                             self.path_weight)
        # haoshi
        candidate_graph_obj.generate_candidate_graph(trajectory, candidate_roads, candidate_points)

        # candidate_graph_obj.show_data()
        return candidate_graph_obj

    def candidate_edge_voting(self, trajectory, candidate_roads, candidate_points, is_show=True):
        n = len(trajectory)
        final_path = []
        lop_seq, f_value_seq = self.find_local_optimal_path_sequence(trajectory, candidate_roads, candidate_points)

        for lop in lop_seq:
            for item in lop:
                for k in range(n - 1):
                    try:
                        self.candidate_graph_obj.edge[f"{k}&{item[k]}|{k + 1}&{item[k + 1]}"].vote += 1
                    except KeyError:
                        pass

        vote = {}
        for i in range(n - 1):
            vote[i] = None

        for edge in self.candidate_graph_obj.edge.values():
            index = int(edge.idx.split('|')[0].split('&')[0])
            if vote[index] is None:
                vote[index] = edge
            elif vote[index].vote < edge.vote:
                vote[index] = edge

        for key, edge in sorted([(key, edge) for key, edge in vote.items()], key=lambda temp: temp[0]):
            final_path.append(edge.fro if edge else None)
            final_path.append(edge.to if edge else None) if key == n - 2 else None

        if is_show:
            # print("所有候选点的局部最优路径：")
            # for i, lop in enumerate(lop_seq):
            #     print(f"采样点{i}: ", lop)
            # print()

            # print("投票结果：")
            # table = PrettyTable(["边", "票数"])
            # for edge in vote.values():
            #     table.add_row([edge.idx, edge.vote] if edge else [None, None])
            # print(table)
            print("最终匹配路径: %s" % final_path)

            # print("路径对应的路段id：", end="")
            #
            # temp = []
            # for point in final_path:
            #     if point is not None:
            #         i, j = map(int, point.split('&'))
            #         temp.append(candidate_roads[i][j])
            #     else:
            #         temp.append(None)
            # print(temp)
            # print("-" * 140)

        return final_path


class Main:
    def __init__(self, mu=5, sigma=25, beta=5, neighbor_num=3):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

        with open("conf.json") as f:
            self.conf = json.load(f)
        self.trajectory_data = dict()
        self.aivmm = AIVMM(self.mu, self.sigma, self.beta, neighbor_num)

        redis_info = {
            "host": "127.0.0.1",
            "password": "",
            "port": 6379,
            "db": 0
        }

        self.r = redis.Redis(**redis_info, decode_responses=True)

    @classmethod
    def calculate_instant_speed(cls, trajectory, rate):
        instantaneous_velocity = []
        for i in range(len(trajectory) - 1):
            lat_0, long_0 = trajectory[i][1], trajectory[i][0]
            lat_1, long_1 = trajectory[i + 1][1], trajectory[i + 1][0]
            distance = geodesic((lat_0, long_0), (lat_1, long_1)).m
            speed = distance / rate * 3.6
            instantaneous_velocity.append(speed)

        return instantaneous_velocity

    def match_candidate(self, trajectory, timestamp, file_name, rate=15):
        # matched_result, match_time = self.aivmm.knn.matched_knn(trajectory)
        url = URL + "match/"
        matched_result, match_time = json.loads(requests.post(url, data=json.dumps(trajectory)).text)
        candidate_distance = []
        candidate_roads = []
        candidate_points = []

        for result_set in matched_result:
            road_temp = []
            point_temp = []
            distance_temp = []

            for road_info in result_set:
                road_temp.append(road_info[0])
                distance_temp.append(road_info[1])
                point_temp.append(road_info[2])

            candidate_distance.append(distance_temp)
            candidate_roads.append(road_temp)
            candidate_points.append(point_temp)

        final_path = self.aivmm.candidate_edge_voting(trajectory, candidate_roads, candidate_points)

        speed_dict = {}
        instant_speed = self.calculate_instant_speed(trajectory, rate)

        for i in range(len(final_path) - 1):
            if final_path[i] is not None and final_path[i + 1]:
                sample_index_pre, candi_index_pre = map(int, final_path[i].split('&'))
                sample_index_cur, candi_index_cur = map(int, final_path[i + 1].split('&'))
                pre_road_id = candidate_roads[sample_index_pre][candi_index_pre]
                cur_road_id = candidate_roads[sample_index_cur][candi_index_cur]

                if pre_road_id not in speed_dict.keys():
                    speed_dict[pre_road_id] = []
                if cur_road_id not in speed_dict.keys():
                    speed_dict[cur_road_id] = []

                if pre_road_id == cur_road_id:
                    speed_dict[pre_road_id].append(instant_speed[i])
                else:
                    speed_dict[cur_road_id].append(instant_speed[i])
                    speed_dict[pre_road_id].append(instant_speed[i])

        for road_id, speed_list in speed_dict.items():
            url = URL + f"update_road_average_speed?road_id={road_id}&average_speed={np.around(np.mean(speed_list), 2)}"
            requests.get(url=url)

        # print("匹配道路的速度值：")
        # table = PrettyTable(["路段id", "时刻", "速度列表", "平均速度"])
        url = URL + "update_road_data/"
        requests.post(url=url,
                      data=json.dumps({"speed_dict": speed_dict, "timestamp": timestamp, "file_name": file_name},
                                      cls=NpEncoder))

        # table.add_row([key, timestamp, value, speed])
        # print(table)
        # print()

        # 画图
        # plt.figure(figsize=(10, 10))
        # plt.scatter([temp[0] for temp in trajectory], [temp[1] for temp in trajectory], color="blue",
        #             label='sample points')
        # x_list = []
        # y_list = []
        # for candi in candidate_points:
        #     for point in candi:
        #         x_list.append(point[0])
        #         y_list.append(point[1])
        #
        # plt.scatter(x_list, y_list, color="#99ff66", alpha=0.5, label="candidate points")
        #
        # for i, candi_roads in enumerate(candidate_roads):
        #     for j, road_id in enumerate(candi_roads):
        #         if i == len(candidate_roads) - 1 and j == len(candi_roads) - 1:
        #             is_label = True
        #         else:
        #             is_label = False
        #         plot_road(road_id, is_label=is_label)
        #
        # x_list = []
        # y_list = []
        # for p in final_path:
        #     if p is not None:
        #         i, j = list(map(int, p.split("&")))
        #         x_list.append(candidate_points[i][j][0])
        #         y_list.append(candidate_points[i][j][1])
        #
        # plt.plot(x_list, y_list, color="red", label="matched path", alpha=0.7)
        # plt.legend(loc=0, ncol=2)
        # plt.show()

    def load_trajectory(self):
        file_path = "data/gps_trajectory"
        self.trajectory_data.clear()

        for file_name in os.listdir(file_path):
            tra_data = pd.read_csv(os.path.join(file_path, file_name), sep=",", encoding="utf-8", iterator=True)
            self.trajectory_data[file_name] = tra_data

    def main(self):
        try:
            while True:
                tra_data = self.r.rpop("trajectory")
                if tra_data:
                    tra_data = json.loads(tra_data)
                    t = time.time()
                    self.match_candidate(tra_data["polyline"], tra_data["timestamp"], tra_data["file_name"])
                    print("匹配用时：", time.time() - t)
                else:
                    break
        except Exception as e:
            print(e)
        # self.load_trajectory()
        # for tra_data in self.trajectory_data.values():
        #     try:
        #         tra_data.get_chunk(self.conf["matched_num"])
        #         while True:
        #             tra_one = tra_data.get_chunk(1)
        #             trip_id = tra_one["TRIP_ID"].values[0]
        #             timestamp = tra_one["TIMESTAMP"].values[0]
        #             try:
        #                 polyline = json.loads(tra_one["POLYLINE"].values[0])
        #                 if len(polyline) < 2:
        #                     raise json.decoder.JSONDecodeError("decoder", "JSONDecodeError", 866)
        #
        #             except json.decoder.JSONDecodeError:
        #                 print(f"{trip_id}: data type error")
        #                 continue
        #             t = time.time()
        #             # try:
        #             self.match_candidate(polyline, timestamp)
        #             self.conf["matched_num"] += 1
        #             with open("conf.json", "w") as f:
        #                 f.write(json.dumps(self.conf))
        #             # except Exception as e:
        #             # print("ERROR: ", e)
        #             self.aivmm.shortest_path_dict.clear()
        #             print("匹配用时: %s\n" % (time.time() - t))
        #     except StopIteration:
        #         print("匹配结束")
        #         break


def plot_road(road_id, is_label=False):
    args = {"color": "#33ccff", "alpha": 0.6}
    if is_label:
        args["label"] = "roads"
    url = URL + f"get_road_nodes?road_id={road_id}"
    road_nodes = json.loads(requests.get(url=url).text)
    plt.plot([temp[0] for temp in road_nodes], [temp[1] for temp in road_nodes], **args)


def load_trajectory():
    file_path = "data/gps_trajectory"
    trajectory = OrderedDict()

    for file in os.listdir(file_path):
        with open(os.path.join(file_path, "shenzhen0.csv"), encoding="utf-8") as f:
            trajectory[file] = [list(map(float, point.strip().split(",")[1:])) for point in f.readlines()[1:]]

    return trajectory


if __name__ == "__main__":
    Main().main()
    # print(DBManager().exec_sql("select file_name,num from finish_flag where id=1"))
