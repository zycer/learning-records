import numpy as np
from get_data import GPSData
from kd_tree import KNN


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


def create_graph(matrix):
    rng = RoadNetworkGraph()

    for edge in matrix:
        idx = edge[0]
        fro = edge[2]
        to = edge[3]
        name = edge[11]

        if fro not in rng.vertex:
            rng.vertex[fro] = Vertex(fro)
        if to not in rng.vertex:
            rng.vertex[to] = Vertex(to)

        from_vertex = rng.vertex[fro]
        to_vertex = rng.vertex[to]

        from_vertex.out += 1
        to_vertex.come += 1

        rng.road_segment.append(
            RoadSegment(idx, fro, to, name, None, None)
        )

    return rng


class AIVMM:
    def __init__(self):
        pass

    # 位置和道路分析
    def gps_observation_probability(self, mu, sigma, x_ij):
        """
        param: x_ij: 候选点i到j的欧氏距离
        GPS点的观测概率
        """
        return np.exp(-(x_ij - mu) ** 2 / 2 * (sigma ** 2))

    def excess_probability(self, distance, omega):
        """
        param: distance: 采样点i-1到采样点i的欧氏距离
        param: omega: 从候选点i-1到i的最短路径长度
        过滤概率函数
        """
        return distance / omega

    def spatial_analysis(self, gop, ep):
        """
        param: gop: GPS点的观测概率
        param: ep: 过度概率
        空间分析函数
        """
        return gop * ep

    def time_analysis(self, hat_v, overline_v):
        """
        param: hat_v: 候选点c_i-1到c_i的最短路径的加权速度限制
        param: hat_v: 沿着候选点c_i-1到c_i之间最短路径行驶的车辆额平均速度
        时间分析函数
        """
        return hat_v / (abs(hat_v - overline_v) + hat_v)

    def road_level_factor(self, vs, vd):
        """
        param: vs: 候选点c_i-1所在道路限速
        param: vd: 候选点c_i所在道路限速
        道路水平系数RLF
        """
        return vs / ((vd - vs) + vs)

    def path_weight(self, fs, ft, rlf):
        """
        param: fs: 空间分析函数
        param: ft: 时间分析函数
        param: rlf: 道路水平因子
        """
        return fs * ft * rlf




# 相互影响分析
