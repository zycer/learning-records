import copy
import json
import math
import os
import random
import numpy as np
from get_data import GPSData
from kd_tree import KNN
from queue import PriorityQueue
from random import uniform, randint
from collections import OrderedDict


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
            self.excess_probability = None

    def __init__(self, check_legitimate_path_func, observation_probability_func, excess_probability_func):
        self.vertex = {}
        self.edge = {}
        self.adjacency_table = {}
        self.check_legitimate_path_func = check_legitimate_path_func
        self.observation_probability_func = observation_probability_func
        self.excess_probability_func = excess_probability_func

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
                    if self.check_legitimate_path_func(point_j, point_k):
                        edge_id = f"{i}&{j}|{i + 1}&{k}"
                        edge = self.Edge(edge_id, f"{i}&{j}", f"{i + 1}&{k}")
                        edge.excess_probability = self.excess_probability_func(trajectory[i], trajectory[i + 1],
                                                                               candidate_roads[i][j],
                                                                               candidate_roads[i][k])
                        self.edge[edge_id] = edge
                        self.adjacency_table[f"{i}&{j}"][f"{i + 1}&{k}"] = edge_id

    def show_data(self):
        for key, value in self.vertex.items():
            print(key, value.idx, value.observation_probability)
        print()
        for key, value in self.edge.items():
            print(value.idx, value.fro, value.to, value.excess_probability)
        print()
        for key, value in self.adjacency_table.items():
            print(key, value)


class RoadNetworkGraph:
    class Vertex:
        def __init__(self, idx, longitude=None, latitude=None):
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

        for vertex_file in self.vertex_data_files:
            try:
                with open(os.path.join(self.vertex_path, vertex_file), encoding="utf-8") as f:
                    vertex_data = f.readlines()[1:]
            except Exception as e:
                print(e)
                continue

            for vertex in vertex_data:
                vertex_attr = vertex.split(",")
                vertex_id = vertex_attr[0]
                longitude = float(vertex_attr[1])
                latitude = float(vertex_attr[2])
                self.vertex[vertex_id] = self.Vertex(vertex_id, longitude, latitude)

        for file in self.road_data_files:
            try:
                with open(os.path.join(self.road_path, file), encoding="utf-8") as f:
                    road_data = f.readlines()[1:]
            except Exception as e:
                print(e)
                continue

            for road in road_data:
                road_attr = road.split("|")
                road_attr_pre = road_attr[0].strip().split(",")
                road_nodes = json.loads(road_attr[1])

                road_id = road_attr_pre[0].strip()
                from_vertex = road_attr_pre[1].strip()
                to_vertex = road_attr_pre[2].strip()
                segment_name = road_attr_pre[3].strip()
                mileage = float(road_attr_pre[4].strip())
                speed_limit = float(road_attr_pre[5])
                average_speed = float(road_attr_pre[6].strip()) if road_attr_pre[5].strip() != '' else -1
                self.matrix.append([road_id, from_vertex, to_vertex,
                                    segment_name, mileage, speed_limit, average_speed, road_nodes])

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
            print(f"{key}: {segment.name}---{segment.mileage}----{segment.road_nodes}")

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
                if math.hypot(x - vertex.longitude, y - vertex.latitude) < 0.05:
                    scope_vertex.append(vertex_id)
            vertex_trajectory_range.append(scope_vertex)

        # 根据轨迹带你附近的节点，查询其出度与入度，并将关联的路段保存[{1:segment_obj, 2: segment_obj}, {...,...},...]
        segment_trajectory_range = []

        for vertexes in vertex_trajectory_range:
            scope_segment = {}
            for vertex_id in vertexes:
                try:
                    temp = self.adjacency_table[vertex_id]
                    for segment in temp.values():
                        if segment.idx not in scope_segment.keys():
                            scope_segment[segment.idx] = segment
                except KeyError:
                    pass

                try:
                    temp = self.inverse_adjacency_table[vertex_id]
                    for segment in temp.values():
                        if segment.idx not in scope_segment.keys():
                            scope_segment[segment.idx] = segment
                except KeyError:
                    pass

            segment_trajectory_range.append(scope_segment)

        return KNN(trajectory, segment_trajectory_range).matched_segments(False)

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
    def __init__(self, graph: RoadNetworkGraph, mu, sigma, beta):
        self.road_graph = graph
        self.mu = mu
        self.beta = beta
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

    def distance_weight(self, point_a, point_b):
        """
        两点之间的距离权重，公式为：exp(dist(pi,pj)^2 / beta^2)
        """
        euclid_distance = self.euclid_distance(point_a, point_b)
        return math.exp(euclid_distance ** 2 / self.beta ** 2)

    def gps_observation_probability(self, candidate_point, sample_point):
        """
        GPS点的观测概率
        param: candidate_point_i: 候选点i
        param: candidate_point_j: 候选点j
        """
        euclid_distance_ij = self.euclid_distance(candidate_point, sample_point)
        return (1 / (math.sqrt(2 * math.pi) * self.sigma)) * math.exp(
            -((euclid_distance_ij - self.mu) ** 2) / (2 * (self.sigma ** 2)))

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
        return euclid_distance / self.get_shortest_path_length(
            self.road_graph.road_segment[pre_road_id].to, self.road_graph.road_segment[cur_road_id].fro)

    def spatial_analysis(self, sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_id, cur_road_id):
        """
        空间分析函数
        param: gop: GPS点的观测概率
        param: ep: 过度概率
        """
        gop = self.gps_observation_probability(candidate_point_cur, sample_point_cur)
        ep = self.excess_probability(sample_point_pre, sample_point_cur, pre_road_id, cur_road_id)
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

    def path_weight(self, sample_point_pre, sample_point_cur, candidate_point_pre,
                    candidate_point_cur, pre_road_id, cur_road_id):
        sa = self.spatial_analysis(sample_point_pre, sample_point_cur, candidate_point_cur, pre_road_id, cur_road_id)
        ta = self.time_analysis(candidate_point_pre, candidate_point_cur)
        rlf = self.road_level_factor(candidate_point_pre, candidate_point_cur)
        return sa * ta * rlf

    # 相互影响分析
    def static_score_matrix(self, trajectory, candidate_roads, candidate_points):
        """
        param: candidate_points: 轨迹的候选点
        静态评分矩阵
        """
        matrix_list = []
        for i in range(len(candidate_points) - 1):
            weight_list = []
            for j, point_j in enumerate(candidate_points[i]):
                for k, point_k in enumerate(candidate_points[i + 1]):
                    weight_list.append(self.path_weight(trajectory[i], trajectory[i + 1], point_j, point_k,
                                                        candidate_roads[i][j], candidate_roads[i + 1][k]))
                    # if i > 0:
                    #     weight_list.append(self.path_weight(trajectory[i - 1], trajectory[i], point_j, point_k,
                    #                                         candidate_roads[i][j], candidate_roads[i][k]))
                    # else:
                    #     weight_list.append(self.path_weight(trajectory[i], trajectory[i+1], point_j, point_k,
                    #                                         candidate_roads[i][j], candidate_roads[i][k]))
                    # weight_list.append(round(random.random(), 2))
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

        print("静态评分矩阵: ")
        print(score_matrix[1:, :])
        print("-------------end-------------")
        return matrix_list

    def distance_weight_matrix(self, trajectory):
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

        print("距离评分矩阵：")
        for m in weight_matrix:
            print(m)
            print()
        print("-------------end-------------")
        return weight_matrix

    def weighted_scoring_matrix(self, trajectory, candidate_roads, candidate_points):
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
                if 1 <= j <= i:
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

        # 打印结果
        print("加权评分矩阵：")
        for matrix in phi_matrix_list:
            print(np.around(matrix, 2))
            print()
        print("-------------end-------------")
        return distance_weight_matrix, phi_list

    def check_legitimate_path(self, point_a, point_b):
        """
        检查两点之间是否可达
        :param point_a: 点a
        :param point_b: 点b
        :return: 是否可达
        """
        result = True if self.get_shortest_path_length(point_a, point_b) else False
        return result

    def find_local_optimal_path(self, candidate_graph, omega_i, phi_i, candi_count, n, i, k):
        """
        获取局部最优路径
        :param candidate_graph: 候选图
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

        for ii in range(i):
            f_ik.append([])
            pre_ik.append([])
            for kk in range(k):
                f_ik[ii].append(-np.inf)
                pre_ik[ii].append(-np.inf)

        for t in range(candi_count[i]):
            f_ik.append(omega_i[i][0] * candidate_graph[f"{0}&{t}"])

        for s in range(candi_count[i]):
            if s != k:
                for t in range(candi_count[i - 1]):
                    phi_i[i][i][t][s] = -np.inf

        for j in range(1, n):
            for s in range(candi_count[j]):
                f_ik[j][s] = max([f_ik[j - 1][t] + phi_i[i][j][t][s] for t in range(candi_count[j - 1])])
                pre_ik[j][s] = max([f_ik[j - 1][t] + phi_i[i][j][t][s] for t in range(candi_count[j - 1])])

        matched_path = []
        c_ik = max([f_ik[n - 1][s] for s in range(candi_count[-1])])
        f_value_cik = max([f_ik[n - 1][s] for s in range(candi_count[-1])])

        for i in range(1, n).__reversed__():
            matched_path.append(c_ik)
            c_ik = pre_ik[c_ik]

        matched_path.append(c_ik)
        matched_path.reverse()
        return matched_path

    def find_local_optimal_path_sequence(self, trajectory, candidate_roads, candidate_points):
        """
        获取局部最优路径序列（所有点的局部最优路径）
        :param trajectory: GPS轨迹点
        :param candidate_roads: 候选点所在的路段id
        :param candidate_points:    候选点
        :return: 列表，局部最优路径序列
        """
        local_optimal_path_sequence = []
        candi_count = [len(points) for points in candidate_points]
        candidate_graph = self.create_candidate_graph(trajectory, candidate_roads, candidate_points)
        distance_weight_matrix, phi_list = self.weighted_scoring_matrix(trajectory, candidate_roads, candidate_points)
        for i, points in enumerate(candidate_points):
            phi_i = phi_list[i]
            for k in range(len(points)):
                local_optimal_path = self.find_local_optimal_path(candidate_graph, distance_weight_matrix[i], phi_i,
                                                                  candi_count, len(trajectory), i, k)
                local_optimal_path_sequence.append(local_optimal_path)

        return local_optimal_path_sequence

    def create_candidate_graph(self, trajectory, candidate_roads, candidate_points):
        candidate_graph = CandidateGraph(self.check_legitimate_path, self.gps_observation_probability,
                                         self.excess_probability)
        candidate_graph.generate_candidate_graph(trajectory, candidate_roads, candidate_points)
        candidate_graph.show_data()
        return candidate_graph.adjacency_table

    def candidate_edge_voting(self, trajectory, candidate_roads, candidate_points):
        n = len(trajectory)
        final_path = []
        lop = self.find_local_optimal_path_sequence(trajectory, candidate_roads, candidate_points)

        # for i in range(n):
        #     for j in +


class Main:
    def __init__(self, mu=5, sigma=25, beta=5):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

        self.road_graph = RoadNetworkGraph()
        self.road_graph.load_road_data()
        self.aivmm = AIVMM(self.road_graph, self.mu, self.sigma, self.beta)

    def match_candidate(self, trajectory):
        matched_result, match_time = self.road_graph.k_nearest_neighbors(trajectory)
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

            # print(candidate_distance)
            # print(candidate_roads)
            # print(candidate_points)
            # print()

        self.aivmm.create_candidate_graph(trajectory, candidate_roads, candidate_points)

        a = self.aivmm.find_local_optimal_path_sequence(trajectory, candidate_roads, candidate_points)
        print(a)


if __name__ == "__main__":
    trajectory_list = [[114.1276453768764, 22.5591123454389], [114.1275677854342, 22.5591122346809]]

    Main().match_candidate(trajectory_list)

    # a = RoadNetworkGraph()
    # a.load_road_data()
    # a.show_graph_data()
    # print("~~~~~~~~~~~~")
    # a.show_adjacency_table()

    #
    # ff = open("data/road_data/lala.csv", "w", encoding="utf-8")
    #
    # with open("data/road_data/new_road_graph.csv", encoding="utf-8") as f:
    #     data = f.readlines()[1:]
    #     for d in data:
    #         d0 = d.split("[")[0]
    #         d1 = d.split("[")[1][:-2]
    #         d2 = d1.split(";")
    #         d3 = [[float(j) for j in i.split(",")] for i in d2]
    #
    #         dd = f"{d0}|{d3}\n"
    #         ff.write(dd)
