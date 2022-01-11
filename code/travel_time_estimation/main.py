import numpy as np
from get_data import GPSData
from kd_tree import KNN
import math


class Vertex:
    def __init__(self, idx):
        self.idx = idx
        self.come = 0
        self.out = 0


class RoadSegment:
    def __init__(self, idx, fro, to, name, speed_limit, road_nodes):
        self.idx = idx
        self.name = name
        self.speed_limit = speed_limit
        self.road_nodes = road_nodes
        self.fro = fro
        self.to = to


class RoadNetworkGraph:
    def __init__(self):
        self.vertex = {}
        self.road_segment = []

    def create_graph(self, matrix):
        for edge in matrix:
            idx = edge[0]
            fro = edge[2]
            to = edge[3]
            name = edge[11]

            if fro not in self.vertex:
                self.vertex[fro] = Vertex(fro)
            if to not in self.vertex:
                self.vertex[to] = Vertex(to)

            from_vertex = self.vertex[fro]
            to_vertex = self.vertex[to]

            from_vertex.out += 1
            to_vertex.come += 1

            self.road_segment.append(
                RoadSegment(idx, fro, to, name, None, None)
            )

    def shortest_path_length(self):
        """
        最短路径长度
        :return:
        """
        pass

    def average_speed_spl(self):
        """
        最短路径上车辆行驶的平均速度
        :return:
        """
        pass

    def weighted_speed_limit_spl(self):
        """
        最短路径上车辆的加权速度限制
        :return:
        """
        pass

    def road_gps_point_located(self, gps_point):
        """
        GPS点所在的道路
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
    def __init__(self, road_graph: RoadNetworkGraph, mu, sigma):
        self.road_graph = road_graph
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

    def get_shortest_path_length(self):
        """
        获取最短路径长度
        :return:
        """
        return self.road_graph.shortest_path_length()

    def get_average_speed_spl(self):
        """
        获取最短路径上车辆行驶的平均速度
        :return:
        """
        return self.road_graph.average_speed_spl()

    def get_weighted_speed_limit_spl(self):
        """
        获取最短路径上车辆的加权速度限制
        :return:
        """
        return self.road_graph.weighted_speed_limit_spl()

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
        return euclid_distance_ij / self.get_shortest_path_length()

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
        ass = self.get_average_speed_spl()
        wsls = self.get_weighted_speed_limit_spl()
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
    


if __name__ == "__main__":
    pass
