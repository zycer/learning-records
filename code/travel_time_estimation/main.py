import numpy as np


# 位置和道路分析
def gps_observation_probability(mu, sigma, x_ij):
    """
    param: x_ij: 候选点i到j的欧氏距离
    GPS点的观测概率
    """
    return np.exp(-(x_ij - mu) ** 2 / 2 * (sigma ** 2))


def excess_probability(distance, omega):
    """
    param: distance: 采样点i-1到采样点i的欧氏距离
    param: omega: 从候选点i-1到i的最短路径长度
    过滤概率函数
    """
    return distance / omega


def spatial_analysis(gop, ep):
    """
    param: gop: GPS点的观测概率
    param: ep: 过度概率
    空间分析函数
    """
    return gop * ep


def time_analysis(hat_v, overline_v):
    """
    param: hat_v: 候选点c_i-1到c_i的最短路径的加权速度限制
    param: hat_v: 沿着候选点c_i-1到c_i之间最短路径行驶的车辆额平均速度
    时间分析函数
    """
    return hat_v / (abs(hat_v - overline_v) + hat_v)


def road_level_factor(vs, vd):
    """
    param: vs: 候选点c_i-1所在道路限速
    param: vd: 候选点c_i所在道路限速
    道路水平系数RLF
    """
    return vs / ((vd - vs) + vs)


def path_weight(fs, ft, rlf):
    """
    param: fs: 空间分析函数
    param: ft: 时间分析函数
    param: rlf: 道路水平因子
    """
    return fs * ft * rlf


# 相互影响分析
