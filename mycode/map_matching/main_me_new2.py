import json
import math
import time
import os

import networkx as nx
import numpy as np
import pandas as pd
import redis

from utils.kd_tree_me import KNN
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils.db_manager import DBManager
from utils.constants import REDIS_INFO
from utils.road_network import BaseRoadNetwork


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
            self.weight = None
            self.vote = 0

    def __init__(self, check_legitimate_path_func, observation_probability_func):
        self.vertex = {}
        self.edge = {}
        self.adjacency_table = {}
        self.check_legitimate_path_func = check_legitimate_path_func
        self.observation_probability_func = observation_probability_func

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
                    # if self.check_legitimate_path_func(self.road_segment[candidate_roads[i][j]],
                    #                                    self.road_segment[candidate_roads[i + 1][k]]):
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


class RoadNetwork(BaseRoadNetwork):
    def __init__(self):
        self.temp_shortest_path = {}
        super().__init__(usage="match")

    def shortest_path(self, source, target):
        """
        dijkstra方法获取最短路径
        :param source:
        :param target:
        :return:
        """
        _path = []
        path_key = f"{source}&{target}"
        if path_key in self.temp_shortest_path.keys():
            return self.temp_shortest_path[path_key]
        try:
            _path = nx.dijkstra_path(self.road_graph, source, target)
        except nx.exception.NetworkXNoPath:
            _path = []
        self.temp_shortest_path[path_key] = _path
        return _path

    def shortest_path_length(self, source, target):
        """
        dijkstra方法获取最短路径长度
        :param source:
        :param target:
        :return:
        """
        path_key = f"{source}&{target}"
        if path_key in self.temp_shortest_path.keys():
            return len(self.temp_shortest_path[path_key])
        try:
            return nx.dijkstra_path_length(self.road_graph, source, target)
        except nx.exception.NetworkXNoPath:
            return 0

    def average_speed_spl(self, source, target):
        """
        最短路径上车辆行驶的平均速度
        :return:
        """
        average_speed_list = []
        shortest_path = self.shortest_path(source, target)
        for road_idx in shortest_path:
            average_speed = self.road_graph.nodes[road_idx]["average_speed"]
            average_speed = average_speed if average_speed else self.road_graph.nodes[road_idx]["free_speed"]
            average_speed_list.append(average_speed)

        return np.around(np.mean(average_speed_list), 6) if average_speed_list else 0

    def weighted_speed_limit_spl(self, source, target):
        """
        最短路径上车辆的加权速度限制
        :return:
        """
        shortest_path = self.shortest_path(source, target)

        return np.around(np.mean([self.road_graph.nodes[road_idx]["free_speed"] for road_idx in shortest_path]),
                         6) if shortest_path else 0

    def road_speed_limit(self, road_idx):
        """
        通过路段id获取该路段限速
        :param road_idx:
        :return:
        """
        return self.road_graph.nodes[road_idx]["free_speed"]


class AIVMM:
    def __init__(self, graph: RoadNetwork, mu, sigma, beta, neighbor_num=3):
        self.road_graph = graph
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.neighbor_num = neighbor_num
        self.candidate_graph_obj = None
        self.shortest_path_dict = {}
        self.knn = KNN(self.road_graph, self.neighbor_num)

    # 位置和道路分析
    @classmethod
    def euclid_distance(cls, source, target):
        """
        欧几里得距离
        :param source: 点A
        :param target: 点B
        :return: 欧几里得距离
        """
        long_1, lat_1 = source
        long_2, lat_2 = target
        return geodesic((lat_1, long_1), (lat_2, long_2)).km

    def distance_weight(self, source, target):
        """
        两点之间的距离权重，公式为：exp(-(dist(pi,pj)^2 / beta^2))
        """
        euclid_distance = self.euclid_distance(source, target)
        return math.exp(-(euclid_distance ** 2 / self.beta ** 2))

    def gps_observation_probability(self, candidate_point, sample_point, candi_dist_cur=0):
        """
        GPS点的观测概率
        param: candidate_point: 候选点
        param: sample_point: 采样点
        公式1: np.exp(-((euclid_distance - self.mu) ** 2) / (2 * (self.sigma ** 2)))
        公式2: (1/math.sqrt(2*np.pi*self.sigma))*np.exp(-((euclid_distance - self.mu) ** 2) / (2 * (self.sigma ** 2)))
        """
        euclid_distance = self.euclid_distance(candidate_point, sample_point)
        # euclid_distance = candi_dist_cur
        return np.exp(-((euclid_distance - self.mu) ** 2) / (2 * (self.sigma ** 2)))

    def get_shortest_path_length(self, source, target):
        """
        获取最短路径长度
        :return:
        """
        return self.road_graph.shortest_path_length(source, target)

    def get_average_speed_spl(self, source, target):
        """
        获取最短路径上车辆行驶的平均速度
        :return:
        """
        return self.road_graph.average_speed_spl(source, target)

    def get_weighted_speed_limit_spl(self, source, target):
        """
        获取最短路径上车辆的加权速度限制
        :return:
        """
        return self.road_graph.weighted_speed_limit_spl(source, target)

    def get_road_speed_limit(self, road_idx):
        """
        获取GPS点所在路段的限速
        :param road_idx:
        :return:
        """
        return self.road_graph.road_speed_limit(road_idx)

    def excess_probability(self, sample_point_pre, sample_point_cur, pre_road_idx, cur_road_idx):
        """
        过滤概率函数
        :param sample_point_pre: 当前采样点前一个采样点
        :param sample_point_cur: 当前采样点
        :param pre_road_idx: 前一个采样点的候选点所在道路的id
        :param cur_road_idx: 当前采样点的候选点所在道路的id
        :return: 两个连续候选点之间的最短路径和直路径的相似性(过度概率)
        """
        euclid_distance = self.euclid_distance(sample_point_pre, sample_point_cur)
        shortest_path_length = self.get_shortest_path_length(pre_road_idx, cur_road_idx)
        return euclid_distance / shortest_path_length if shortest_path_length else 0

    def spatial_analysis(self, sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_idx, cur_road_idx, candi_dist_cur):
        """
        空间分析函数
        param: gop: GPS点的观测概率
        param: ep: 过度概率
        """
        gop = self.gps_observation_probability(candidate_point_cur, sample_point_cur, candi_dist_cur)
        ep = self.excess_probability(sample_point_pre, sample_point_cur, pre_road_idx, cur_road_idx)
        return gop * ep

    def time_analysis(self, source, target):
        """
        时间分析函数
        param: source: 候选点i
        param: target: 沿着候选点j
        """
        ass = self.get_average_speed_spl(source, target)
        wsls = self.get_weighted_speed_limit_spl(source, target)
        return ass / (abs(ass - wsls) + ass) if abs(ass - wsls) + ass else 0

    def road_level_factor(self, pre_road_idx, cur_road_idx):
        """
        道路水平系数RLF
        param: vs: 候选点c_i-1所在道路限速
        param: vd: 候选点c_i所在道路限速
        """

        segment_i_speed_limits = self.get_road_speed_limit(pre_road_idx)
        segment_j_speed_limits = self.get_road_speed_limit(cur_road_idx)
        return segment_i_speed_limits / ((segment_j_speed_limits - segment_i_speed_limits) + segment_i_speed_limits)

    def path_weight(self, sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_idx, cur_road_idx, candi_dist_cur):
        sa = self.spatial_analysis(sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_idx, cur_road_idx, candi_dist_cur)
        ta = self.time_analysis(pre_road_idx, cur_road_idx)     # ta误差大
        rlf = self.road_level_factor(pre_road_idx, cur_road_idx)
        return ta * sa * rlf

    # 相互影响分析
    def static_score_matrix(self, trajectory, candidate_roads, candidate_points, candidate_distance, is_show=False):
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
                                                        candidate_roads[i][j], candidate_roads[i + 1][k], candidate_distance[i+1][k]))
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

    def weighted_scoring_matrix(self, trajectory, candidate_roads, candidate_points, candidate_distance, is_show=False):
        """
        param: trajectory: GPS轨迹点
        param: candidate_points: GPS轨迹点对应的候选点
        加权得分矩阵
        """
        static_score_matrix = self.static_score_matrix(trajectory, candidate_roads, candidate_points, candidate_distance)
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

    def check_legitimate_path(self, source, target):
        """
        检查两点之间是否可达
        :param source: 点a
        :param target: 点b
        :return: 是否可达
        """
        return True if self.get_shortest_path_length(source, target) else False

    def new_find_lops(self, omega_i, phi_i, n, candi_count):
        f_values = np.zeros((n, candi_count[0]))
        for i in range(n):
            if i == 0:
                for t in range(candi_count[i]):
                    temp_value = omega_i[0, 0] * self.candidate_graph_obj.candidate_graph.nodes[f"{0}&{t}"][
                        "observation_probability"]
                    f_values[0, t] = temp_value
            else:
                for s in range(candi_count[i]):
                    f_values[i, s] = np.max([f_values[i-1][t]+phi_i[i-1][t,s] for t in range(candi_count[i])])

        lop = [np.argmax(values) for values in f_values]
        return lop

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
                temp = [f_ik[j - 1][t] + phi_i[j - 1][t, s] for t in range(candi_count[j - 1])]
                f_ik[j][s] = max(temp)
                pre_ik[j][s] = np.argmax(temp)

        matched_path = []

        _temp = [f_ik[n - 1][s] for s in range(candi_count[-1])]
        c = np.argmax(_temp)
        f_value_cik = max(_temp)

        for i in range(1, n).__reversed__():
            matched_path.append(c)
            c = pre_ik[i][c]

        matched_path.append(c)
        matched_path.reverse()

        return matched_path, f_value_cik

    def find_local_optimal_path_sequence(self, trajectory, candidate_roads, candidate_points, candidate_distance):
        """
        获取局部最优路径序列（所有点的局部最优路径）
        :param trajectory: GPS轨迹点
        :param candidate_roads: 候选点所在的路段id
        :param candidate_points:    候选点
        :return: 列表，局部最优路径序列
        """
        t_time = 0
        local_optimal_path_sequence = []
        f_value_sequence = []

        candi_count = [len(points) for points in candidate_points]
        # haoshi
        self.candidate_graph_obj = self.create_candidate_graph(trajectory, candidate_roads, candidate_points, candidate_distance)
        distance_weight_matrix, phi_list = self.weighted_scoring_matrix(trajectory, candidate_roads, candidate_points, candidate_distance)

        for i, points in enumerate(candidate_points):
            phi_i = phi_list[i]
            lop_i = []
            f_value_i = []
            for k in range(len(points)):
                # t = time.time()
                local_optimal_path, f_value = self.find_local_optimal_path(distance_weight_matrix[i], phi_i,
                                                               candi_count, len(trajectory), i, k)
                # t_time += time.time() - t
                # local_optimal_path = self.new_find_lops(distance_weight_matrix[i], phi_i, len(trajectory), candi_count)
                lop_i.append(local_optimal_path)
                # f_value_i.append(f_value)
            local_optimal_path_sequence.append(lop_i)
            f_value_sequence.append(f_value_i)

        # print("total_time:", t_time)
        return local_optimal_path_sequence, f_value_sequence, t_time

    def create_candidate_graph(self, trajectory, candidate_roads, candidate_points, candidate_distance):
        candidate_graph_obj = CandidateGraph(self.check_legitimate_path, self.gps_observation_probability)
        # haoshi
        candidate_graph_obj.generate_candidate_graph(trajectory, candidate_roads, candidate_points)

        # candidate_graph_obj.show_data()
        return candidate_graph_obj

    def candidate_edge_voting(self, trajectory, candidate_roads, candidate_points, candidate_distance, is_show=True):
        n = len(trajectory)
        final_path = []
        lop_seq, f_value_seq, t_time = self.find_local_optimal_path_sequence(trajectory, candidate_roads, candidate_points, candidate_distance)
        for i in lop_seq:
            print(i)
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

        # if is_show:
        # print("所有候选点的局部最优路径：")
        # for i, lop in enumerate(lop_seq):
        #     print(f"采样点{i}: ", lop)
        # print()

        # print("投票结果：")
        # table = PrettyTable(["边", "票数"])
        # for edge in vote.values():
        #     table.add_row([edge.idx, edge.vote] if edge else [None, None])
        # print(table)
        # print("最终匹配路径: %s" % final_path)

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

        return final_path, t_time


class Main:
    def __init__(self, mu=5, sigma=25, beta=5, neighbor_num=4):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.trajectory_data = dict()
        print("加载路网数据...")
        self.road_graph = RoadNetwork()
        print("加载完成: ")
        print("道路数：%s, 路口数：%s" % (len(self.road_graph.road_graph.nodes), len(self.road_graph.road_graph.edges)))
        self.db_handler = DBManager()
        self.aivmm = AIVMM(self.road_graph, self.mu, self.sigma, self.beta, neighbor_num)

        self.r = redis.Redis(**REDIS_INFO, decode_responses=True)

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

    def match_candidate(self, trajectory, timestamp, is_show=False, rate=15):
        matched_result, match_time = self.aivmm.knn.matched_knn(trajectory)

        candidate_distance = []
        candidate_roads = []
        candidate_points = []
        candidate_segments = []

        for result_set in matched_result:
            road_temp = []
            point_temp = []
            distance_temp = []
            segment_temp = []

            for road_info in result_set:
                road_temp.append(road_info[0])
                distance_temp.append(road_info[1])
                point_temp.append(road_info[2])
                segment_temp.append(road_info[3])

            candidate_distance.append(distance_temp)
            candidate_roads.append(road_temp)
            candidate_points.append(point_temp)
            candidate_segments.append(segment_temp)

        final_path, t_time = self.aivmm.candidate_edge_voting(trajectory, candidate_roads, candidate_points, candidate_distance)

        speed_dict = {}
        instant_speed = self.calculate_instant_speed(trajectory, rate)

        for i in range(len(final_path) - 1):
            if final_path[i] is not None and final_path[i + 1]:
                sample_index_pre, candi_index_pre = map(int, final_path[i].split('&'))
                sample_index_cur, candi_index_cur = map(int, final_path[i + 1].split('&'))
                pre_road_id = candidate_roads[sample_index_pre][candi_index_pre].item()
                cur_road_id = candidate_roads[sample_index_cur][candi_index_cur].item()

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
            self.road_graph.road_graph.nodes[road_id]["average_speed"] = np.around(np.mean(speed_list), 2)
        #
        # matched_info = {"road_info": speed_dict, "timestamp": timestamp}
        # self.r.lpush("matched_result", json.dumps(matched_info))

        # print("匹配道路的速度值：")
        # table = PrettyTable(["路段id", "时刻", "速度列表", "平均速度"])
        # for key, value in speed_dict.items():
        #     speed = np.around(np.mean(value), 2).item()
        #     timestamp = timestamp.item() if isinstance(timestamp, np.int64) else timestamp
        #     key = key.item()
        #     road_obj = self.db_handler.exec_sql(f"SELECT * FROM history_road_data WHERE road_id='{key}'")
        #     if road_obj:
        #         history = json.loads(road_obj[0][1])
        #         history[str(timestamp)] = speed
        #         average_speed = sum(history.values()) / len(history)
        #         self.db_handler.exec_sql(
        #             f"UPDATE history_road_data set history='{json.dumps(history)}',average_speed={average_speed} WHERE road_id='{key}'")
        #     else:
        #         history = {str(timestamp): speed}
        #         self.db_handler.exec_sql(
        #             f"INSERT INTO history_road_data VALUES ('{key}','{json.dumps(history)}',{speed}, '{self.road_graph.road_segment[key].fro}','{self.road_graph.road_segment[key].to}')")
        # table.add_row([key, timestamp, value, speed])

        # print(table)
        # print()

        # 画图
        if is_show:
            plt.figure(figsize=(10, 10))
            plt.scatter([temp[0] for temp in trajectory], [temp[1] for temp in trajectory], color="blue",
                        label='sample points')
            x_list = []
            y_list = []
            for candi in candidate_points:
                for point in candi:
                    x_list.append(point[0])
                    y_list.append(point[1])

            plt.scatter(x_list, y_list, color="#99ff66", alpha=0.5, label="candidate points")

            for i, candi_roads in enumerate(candidate_roads):
                for j, road_id in enumerate(candi_roads):
                    if i == len(candidate_roads) - 1 and j == len(candi_roads) - 1:
                        is_label = True
                    else:
                        is_label = False
                    plot_road(self.road_graph.road_graph.nodes[road_id], is_label=is_label)

            x_list = []
            y_list = []
            for p in final_path:
                if p is not None:
                    i, j = list(map(int, p.split("&")))
                    x_list.append(candidate_points[i][j][0])
                    y_list.append(candidate_points[i][j][1])

            plt.plot(x_list, y_list, color="red", label="matched path", alpha=0.7)
            plt.legend(loc=0, ncol=2)
            plt.show()
        return candidate_points, final_path, candidate_segments, t_time

    def main(self):
        candidate_data = {"timestamp": [], "trajectory": [], "candidate_points": [], "final_path": [],
                          "candidate_segments": []}
        trajectory_data = self.r.lrange("trajectory", 0, -1)
        count = 0
        for data in trajectory_data:
            tra_data = json.loads(data)
            count += len(tra_data["polyline"])
        print(count)
        lop_t_time = 0
        total_time = 0
        try:
            for index, tra_data in enumerate(trajectory_data):
                if tra_data:
                    tra_data = json.loads(tra_data)
                    len_tra_data = len(tra_data["polyline"])
                    if len_tra_data < 3:
                        print("跳过数据: ", tra_data["timestamp"])
                        # self.db_handler.exec_sql("UPDATE finish_flag set num=num+1 WHERE file_name='train.csv'")
                        continue
                    elif len_tra_data <= 100:
                        t = time.time()
                        result = self.match_candidate(tra_data["polyline"], tra_data["timestamp"])
                        candidate_points, final_path, candidate_segments, t_time = result
                        # lop_t_time += t_time
                        temp_time = time.time() - t
                        total_time += temp_time
                        print("匹配用时：", temp_time)
                        update_matched_data(candidate_data, tra_data["polyline"], tra_data["timestamp"],
                                            candidate_points, final_path, candidate_segments)

                    else:
                        while len_tra_data // 100:
                            temp_tra = tra_data["polyline"][:100]
                            t = time.time()
                            result = self.match_candidate(temp_tra, tra_data["timestamp"])
                            candidate_points, final_path, candidate_segments, t_time = result
                            # lop_t_time += t_time
                            temp_time = time.time() - t
                            total_time += temp_time
                            print("分解匹配用时：", temp_time)
                            tra_data["polyline"] = tra_data["polyline"][100:]
                            len_tra_data = len(tra_data["polyline"])

                            update_matched_data(candidate_data, tra_data["polyline"], tra_data["timestamp"],
                                                candidate_points, final_path, candidate_segments)

                        if len(tra_data["polyline"]) > 2:
                            t = time.time()
                            result = self.match_candidate(tra_data["polyline"], tra_data["timestamp"])
                            candidate_points, final_path, candidate_segments, t_time = result
                            # lop_t_time += t_time
                            temp_time = time.time() - t
                            total_time += temp_time
                            print("分解匹配用时：", temp_time)
                            update_matched_data(candidate_data, tra_data["polyline"], tra_data["timestamp"],
                                                candidate_points, final_path, candidate_segments)

                    if len(candidate_data["timestamp"]) > 0:
                        save_matched_data(candidate_data, index)
                        candidate_data = {"timestamp": [], "trajectory": [], "candidate_points": [], "final_path": [],
                                          "candidate_segments": []}

                    self.road_graph.temp_shortest_path.clear()
                else:
                    print("所有数据匹配完成")
                    break
        except Exception:
            raise
            # self.r.rpush("trajectory", json.dumps(tra_data))
            # raise Exception
        print("总时间：", total_time)
        # print("t_time:", lop_t_time)


def save_matched_data(candidate_data: dict, index):
    db_handler = DBManager()
    file_name = "candidates_me_new.csv"
    if index == 0 and os.path.exists(f"data/candidate_data/{file_name}"):
        os.remove(f"data/candidate_data/{file_name}")
    header = None if file_name in os.listdir("data/candidate_data") else \
        ["timestamp", "trajectory", "candidate_points", "final_path", "candidate_segments"]

    data_frame = pd.DataFrame({"timestamp": candidate_data["timestamp"],
                               "trajectory": candidate_data["trajectory"],
                               "candidate_points": candidate_data["candidate_points"],
                               "final_path": candidate_data["final_path"],
                               "candidate_segments": candidate_data["candidate_segments"]})
    data_frame.to_csv(f"data/candidate_data/{file_name}", index=False, sep=',', mode="a+", header=header)
    db_handler.exec_sql(
        f"UPDATE finish_flag set num=num+{len(candidate_data['timestamp'])} WHERE file_name='train.csv'")
    print("save matched data")


def update_matched_data(candidate_data, trajectory, timestamp, candidate_points, final_path, candidate_segments):
    candidate_data["timestamp"].append(timestamp)
    candidate_data["trajectory"].append(trajectory)
    candidate_data["candidate_points"].append(candidate_points)
    candidate_data["final_path"].append(final_path)
    candidate_data["candidate_segments"].append(candidate_segments)
    return candidate_data


def plot_road(road_obj, is_label=False):
    args = {"color": "#33ccff", "alpha": 0.6}
    if is_label:
        args["label"] = "roads"
    plt.plot([temp[0] for temp in road_obj.geometry], [temp[1] for temp in road_obj.geometry], **args)


if __name__ == "__main__":
    Main(neighbor_num=3).main()

    # KNN算法比较
    # r = redis.Redis(**REDIS_INFO, decode_responses=True)
    # trajectory_data = r.lrange("trajectory", 0, -1)
    # road_graph = RoadNetwork()
    # knn = KNN(road_graph, 6)
    #
    # total_time = 0
    # for data in trajectory_data:
    #     data = json.loads(data)
    #     res, end_time = knn.matched_knn(data["polyline"])
    #     total_time += end_time
    # print(total_time)

    # 最短路径算法比较
    # total_time = 0
    # for data in trajectory_data:
    #     data = json.loads(data)
    #     matched_result, match_time = knn.matched_knn(data["polyline"])
    #
    #     candidate_roads = []
    #     candidate_points = []
    #
    #     for result_set in matched_result:
    #         road_temp = []
    #         point_temp = []
    #
    #         for road_info in result_set:
    #             road_temp.append(road_info[0])
    #             point_temp.append(road_info[2])
    #
    #         candidate_roads.append(road_temp)
    #         candidate_points.append(point_temp)
    #
    #     for i in range(len(candidate_points) - 1):
    #         for j, point_j in enumerate(candidate_points[i]):
    #             for k, point_k in enumerate(candidate_points[i + 1]):
    #                 t = time.time()
    #                 road_graph.shortest_path(candidate_roads[i][j], candidate_roads[i + 1][k])
    #                 total_time += time.time() - t
    #
    # print(total_time)