import math
import time
import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import numpy as np
from prettytable import PrettyTable


class KNN:
    def __init__(self, trajectory, roads, neighbor_num=3):
        """
        param: trajectory: GPS轨迹点列表 [[x1, y1], [x2, y2],...]
        param: roads: 路网中所有道路
        param: neighbor_num: k个邻居

        self.segment_id_list: 与列表self.segment_lines对应顺序的道路真实id
        [[road_id1,road_id1,road_id2,...], [road_id3,...]]
        """
        self.trajectory = trajectory
        self.roads = roads
        self.neighbor_num = neighbor_num
        self.segment_lines = []
        self.segment_id_list = []
        self.r = 6367

        self.roads_segments = []
        self.roads_segments_id = []

        for road_id, road_obj in roads[0].items():
            for i in range(len(road_obj.road_nodes) - 1):
                self.roads_segments.append([[road_obj.road_nodes[i][0], road_obj.road_nodes[i][1]],
                                 [road_obj.road_nodes[i + 1][0], road_obj.road_nodes[i + 1][1]]])
                self.roads_segments_id.append(road_id)

        road_segments_data = self.change_data(np.concatenate(self.roads_segments))
        self.tree = spatial.cKDTree(road_segments_data[:, 2:5])
        self.true_road_index = tuple(itertools.chain.from_iterable(
            [itertools.repeat(i, road) for i, road in enumerate(list(map(len, self.roads_segments)))]
        ))

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
            return calculate(longitude, latitude)

        return result

    @classmethod
    def mercator2wgs84(cls, mercator):
        def calculate(xx, yy):
            long = float(xx) / 20037508.34 * 180
            lat = float(yy) / 20037508.34 * 180
            lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
            return [long, lat]

        result = []
        if mercator is None or len(mercator) == 0:
            return result
        elif isinstance(mercator[0], list):
            for x, y in mercator:
                result.append(calculate(x, y))
        else:
            x, y = mercator
            return calculate(x, y)

        return result

    @classmethod
    def generate_equation(cls, **kwargs):
        if "points" in kwargs.keys():
            first_point, second_point = kwargs["points"]
            a = second_point[1] - first_point[1]
            b = first_point[0] - second_point[0]
            c = second_point[0] * first_point[1] - first_point[0] * second_point[1]
            try:
                k = -1 * a / b
                b = -1 * c / b
            except ZeroDivisionError:
                return None, None, None

        else:
            point = kwargs["point"]
            k = kwargs["k"]
            # 垂直于x轴
            if k is None:
                return None, None, None
            else:
                b = point[1] - k * point[0]

        def equation(x):
            return k * x + b

        return equation, k, b

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

    @classmethod
    def is_mid(cls, point, segment_mercator):
        segment_iter = zip(segment_mercator[0], segment_mercator[1])
        for num, segment in enumerate(segment_iter):
            if sorted(segment)[0] <= point[num] <= sorted(segment)[1]:
                continue
            else:
                return False
        return True

    def generate_candidate_point(self, segment, point):
        point_mercator: list = self.wgs842mercator(point)
        segment_mercator = self.wgs842mercator(segment)
        segment_equation, k0, b0 = self.generate_equation(**{"points": segment_mercator})
        if segment_equation is not None:
            vertical_equation, k1, b1 = self.generate_equation(**{"point": point_mercator, "k": None if k0 == 0 else -1 / k0})
            if vertical_equation is not None:
                vertical_point = [(b1 - b0) / (k0 - k1), vertical_equation((b1 - b0) / (k0 - k1))]
            else:
                vertical_point = [segment_mercator[0][0], point_mercator[1]]
        # 垂直于x轴
        else:
            vertical_point = [segment_mercator[0][0], point_mercator[1]]

        if not self.is_mid(vertical_point, segment_mercator):
            distance_1 = math.hypot(point_mercator[0] - segment_mercator[0][0],
                                    point_mercator[1] - segment_mercator[0][1])

            distance_2 = math.hypot(point_mercator[0] - segment_mercator[1][0],
                                    point_mercator[1] - segment_mercator[1][1])

            candidate_point = segment[0] if distance_1 < distance_2 else segment[1]

        else:
            candidate_point = self.mercator2wgs84(vertical_point)

        return self.is_mid(vertical_point, segment_mercator), tuple(candidate_point)

    def matched_knn(self, is_plot=True):
        """
        匹配k个邻居
        param: is_plot: 是否画图显示
        matched: [{(road_id, distance,[long, lat]), (road_id, distance), [long, lat]...},...},...]
        """
        matched = []
        start_time = time.time()

        for num, segment_line in enumerate(self.segment_lines):
            last_matched = set()
            k = self.neighbor_num
            lines = self.change_data(np.concatenate(segment_line))
            tree = spatial.cKDTree(lines[:, 2:5])
            lines_ix = tuple(itertools.chain.from_iterable(
                [itertools.repeat(i, road) for i, road in enumerate(list(map(len, segment_line)))]
            ))
            trajectory = self.change_data(np.concatenate([[self.trajectory[num]]]))
            t2 = time.time()
            for ii in range(1000):
                tra = self.change_data(np.concatenate([[self.trajectory[num]]]))
                distance, roads_idx = tree.query(tra[:, 2:5], k=100)
            print("查找：", time.time()-t2)
            # print("查找结果：", roads_idx)

            exit()

            print("pipei: ", self.trajectory[num])

            while len(segment_line) >= k:
                trajectory = self.change_data(np.concatenate([[self.trajectory[num]]]))
                distance, roads_idx = tree.query(trajectory[:, 2:5], k=k)
                distance = self.dist_to_arc_length(distance)
                match_dict = OrderedDict()
                print("while")
                for index, segment_id in enumerate(itemgetter(*roads_idx[0])(lines_ix)):
                    # temp = [segment[0] for segment in match_set]
                    is_mid, candidate_point = self.generate_candidate_point(segment_line[segment_id], self.trajectory[num])

                    if self.segment_id_list[num][segment_id] not in match_dict:
                        match_dict[self.segment_id_list[num][segment_id]] = (distance[0][index], candidate_point)
                    elif is_mid:
                        del match_dict[self.segment_id_list[num][segment_id]]
                        match_dict[self.segment_id_list[num][segment_id]] = (distance[0][index], candidate_point)
                    else:
                        pass

                if len(match_dict) == self.neighbor_num:
                    match_set = [(key, value[0], value[1]) for key, value in match_dict.items()]
                    matched.append(match_set)
                    break
                else:
                    k += 1

                last_matched = match_dict

            if k > len(segment_line):
                match_set = [(key, value[0], value[1]) for key, value in last_matched.items()]
                matched.append(match_set)

        print(f"候选点匹配用时<{round(time.time() - start_time, 6)}>秒，匹配结果: \n")
        for i, result in enumerate(matched):
            print(f"采样点{i}: ")
            table = PrettyTable(["路段ID", "距离", "经度", "纬度"])
            for res in result:
                table.add_row([res[0], res[1], res[2][1], res[2][1]])
            print(table)
            print()

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
