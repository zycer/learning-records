import math
import time

import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import itertools
import numpy as np


class KNN:
    def __init__(self, trajectory, roads, neighbor_num=2):
        """
        param: trajectory: GPS轨迹点列表 [[x1, y1], [x2, y2],...]
        param: segments: GPS轨迹点附近所有路段列表 [{id1: segment_obj1, id2: segment_obj2,...}, {idx: segment_obj,...},...]
        param: k: k个邻居

        self.segment_lines: 每条路被多个road_node点表示，将所有路解析成嵌套列表
        [[[[node_longitude1, node_latitude1], [node_longitude2, node_latitude2],...],...],...]

        self.segment_id_list: 与列表self.segment_lines对应顺序的道路真实id
        [[road_id1,road_id1,road_id2,...], [road_id3,...]]
        """
        self.trajectory = trajectory
        self.segments = roads
        self.neighbor_num = neighbor_num
        self.segment_lines = []
        self.segment_id_list = []
        self.r = 6367

    @classmethod
    def wgs842mercator(cls, wgs84):
        def calculate(long, lat):
            x = float(long) * 20037508.34 / 180
            y = math.log(math.tan((90 + float(lat)) * math.pi / 360)) / (math.pi / 180)
            y = y * 20037508.34 / 180
            return [x, y]

        result = []
        if wgs84 is None or len(wgs84) == 0:
            return result
        elif isinstance(wgs84[0], list):
            for longitude, latitude in wgs84:
                result.append(calculate(longitude, latitude))
        else:
            longitude, latitude = wgs84
            result.append(calculate(longitude, latitude))

        return result

    @classmethod
    def mercator2wgs84(cls, mercator):
        def calculate(xx, yy):
            long = float(x) / 20037508.34 * 180
            lat = float(y) / 20037508.34 * 180
            lat = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
            return [long, lat]

        result = []
        if mercator is None or len(mercator) == 0:
            return result
        elif isinstance(mercator[0], list):
            for x, y in mercator:
                result.append(calculate(x, y))
        else:
            x, y = mercator
            result.append(calculate(x, y))

        return result

    @classmethod
    def generate_equation(cls, **kwargs):
        if "points" in kwargs.keys():
            first_point, second_point = kwargs["points"]
            a = second_point[1] - first_point[1]
            b = first_point[0] - second_point[0]
            c = second_point[0] * first_point[1] - first_point[0] * second_point[1]
            k = -1 * a / b
            b = -1 * c / b

        else:
            point = kwargs["point"]
            k = kwargs["k"]
            b = point[1] - k * point[0]

        def equation(x):
            return k * x + b

        return equation

    def change_data(self, data):
        """
        经纬度坐标转换
        """
        phi = np.deg2rad(data[:, 1])  # LAT
        theta = np.deg2rad(data[:, 0])  # LON
        data = np.c_[
            data, self.r * np.cos(phi) * np.cos(theta), self.r * np.cos(phi) * np.sin(theta), self.r * np.sin(phi)]
        return data

    def dist_to_arc_length(self, chord_length):
        """
        弧长转换为球面距离
        """
        central_angle = 2 * np.arcsin(chord_length / (2.0 * self.r))
        return self.r * central_angle

    def data_pretreatment(self):
        """
        数据预处理
        """
        segment_lines = []
        segment_id_list = []
        for segment in self.segments:
            lines = []
            segment_ids = []
            for segment_id, segment_obj in segment.items():
                for i in range(len(segment_obj.road_nodes) - 1):
                    lines.append([[segment_obj.road_nodes[i][0], segment_obj.road_nodes[i][1]],
                                  [segment_obj.road_nodes[i + 1][0], segment_obj.road_nodes[i + 1][1]]])
                    segment_ids.append(segment_id)

            segment_id_list.append(segment_ids)
            segment_lines.append(lines)

        self.segment_lines = segment_lines
        self.segment_id_list = segment_id_list

    def generate_candidate_point(self, segment, point):
        segment_mercator = self.wgs842mercator(segment)
        point_mercator = self.wgs842mercator(point)
        segment_equation = self.generate_equation(**{"points": segment_mercator})
        vertical_bisector_equation = self.generate_equation(**{"point": point, "k":})


    def matched_segments(self, is_plot=True):
        """
        匹配k个邻居
        param: is_plot: 是否画图显示
        matched: [{(road_id, distance), (road_id, distance),...}, {(road_id, distance), (road_id, distance),...},...]
        """
        matched = []
        self.data_pretreatment()
        start_time = time.time()

        for num, segment_line in enumerate(self.segment_lines):
            last_matched = set()
            k = self.neighbor_num
            lines = self.change_data(np.concatenate(segment_line))
            tree = spatial.cKDTree(lines[:, 2:5])
            lines_ix = tuple(itertools.chain.from_iterable(
                [itertools.repeat(i, road) for i, road in enumerate(list(map(len, segment_line)))]
            ))

            while len(segment_line) >= k:
                trajectory = self.change_data(np.concatenate([[self.trajectory[num]]]))
                distance, roads_idx = tree.query(trajectory[:, 2:5], k=k)
                distance = self.dist_to_arc_length(distance)
                match_set = set()
                for index, segment_id in enumerate(itemgetter(*roads_idx[0])(lines_ix)):
                    temp = [segment[0] for segment in match_set]
                    if self.segment_id_list[num][segment_id] not in temp:
                        # todo 增加候选点
                        self.generate_candidate_point(segment_line[segment_id], self.trajectory[num])
                        match_set.add((self.segment_id_list[num][segment_id], distance[0][index]))

                if len(match_set) == self.neighbor_num:
                    matched.append(match_set)
                    break
                else:
                    k += 1

                last_matched = match_set

            if k > len(segment_line):
                matched.append(last_matched)

        print("匹配结果：", matched)
        print("匹配用时：", time.time() - start_time)

        if is_plot:
            self.plot_result()
        return matched, time.time() - start_time

    def plot_result(self):
        for i, segment in enumerate(self.segment_lines):
            plt.figure(figsize=(20, 20), dpi=80)
            for j, lines in enumerate(segment):
                plt.plot([lines[0][0], lines[1][0]], [lines[0][1], lines[1][1]],
                         label=f"{j}-{self.segment_id_list[i][j]}")

            plt.scatter(self.trajectory[i][0], self.trajectory[i][1])
            plt.legend(loc=0, ncol=2)
            plt.show()
