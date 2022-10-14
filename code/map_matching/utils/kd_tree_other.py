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
    def __init__(self, roads, neighbor_num=2):
        """
        param: trajectory: GPS轨迹点列表 [[x1, y1], [x2, y2],...]
        param: roads: 路网中所有道路
        param: neighbor_num: k个邻居

        self.segment_id_list: 与列表self.segment_lines对应顺序的道路真实id
        [[road_id1,road_id1,road_id2,...], [road_id3,...]]
        """
        self.roads = roads
        self.neighbor_num = neighbor_num
        # self.segment_lines = []
        # self.segment_id_list = []
        self.r = 6367

        self.roads_segments = []
        self.roads_segments_id = []

        # qu diao 0
        for road_id, road_obj in roads.items():
            for i in range(len(road_obj.road_nodes) - 1):
                self.roads_segments.append([[road_obj.road_nodes[i][0], road_obj.road_nodes[i][1]],
                                            [road_obj.road_nodes[i + 1][0], road_obj.road_nodes[i + 1][1]]])
                self.roads_segments_id.append(road_id)

        road_segments_data = np.concatenate(self.roads_segments)
        self.tree = spatial.cKDTree(road_segments_data)
        self.segments_ix = tuple(itertools.chain.from_iterable(
            [itertools.repeat(i, road) for i, road in enumerate(list(map(len, self.roads_segments)))]
        ))


    def generate_candidate_point(self, roads_idx, trajectory):
        candidate_points = []
        candidate_segment = []

        for num, point in enumerate(trajectory):
            one_tra_candi_point = []
            one_tra_candi_segment = []
            matched_segments_ix = itemgetter(*roads_idx[num])(self.segments_ix)
            for seg_id in matched_segments_ix:
                segment = self.roads_segments[seg_id]
                candidate_point = ((segment[0][0] + segment[1][0]) / 2, (segment[0][1] + segment[1][1]) / 2)

                one_tra_candi_point.append(candidate_point)
                one_tra_candi_segment.append(tuple(segment))
            candidate_segment.append(one_tra_candi_segment)
            candidate_points.append(one_tra_candi_point)

        return candidate_points, candidate_segment

    def matched_knn(self, trajectory, is_plot=False, is_show=False):
        """
        匹配k个邻居
        param: is_plot: 是否画图显示
        matched: [{(road_id, distance,[long, lat]), (road_id, distance), [long, lat]...},...},...]
        """
        matched = []
        start_time = time.time()
        change_trajectory_data = np.concatenate([trajectory])
        distance, segments_ix = self.tree.query(change_trajectory_data, k=self.neighbor_num)
        candidate_points, candidate_segment = self.generate_candidate_point(segments_ix, trajectory)

        for num in range(len(trajectory)):
            one_tra_matched = list()
            matched_segments_ix = itemgetter(*segments_ix[num])(self.segments_ix)
            for i, seg_ix in enumerate(matched_segments_ix):
                one_tra_matched.append((self.roads_segments_id[seg_ix], distance[num][i], candidate_points[num][i],
                                        candidate_segment[num][i]))
            matched.append(one_tra_matched)

        if is_show:
            print(f"候选点匹配用时<{round(time.time() - start_time, 6)}>秒")
            # for i, result in enumerate(matched):
            #     print(f"采样点{i}: ")
            #     table = PrettyTable(["路段ID", "距离", "经度", "纬度"])
            #     for res in result:
            #         table.add_row([res[0], res[1], res[2][1], res[2][1]])
            #     print(table)
            #     print()

        if is_plot:
            self.plot_result(trajectory)
        return matched, time.time() - start_time

    def plot_result(self, trajectory):
        for i, segment in enumerate(self.roads_segments):
            plt.figure(figsize=(20, 20), dpi=80)
            for j, lines in enumerate(segment):
                plt.plot([lines[0][0], lines[1][0]], [lines[0][1], lines[1][1]],
                         label=f"{j}-{self.roads_segments_id[j]}")

            plt.scatter(trajectory[i][0], trajectory[i][1])
            plt.legend(loc=0, ncol=2)
            plt.show()

# knn = KNN([[-8.602785, 41.145705], [-8.12345, 41.134553]], 123)
# print(knn.change_data(np.concatenate([[[-8.602785, 41.145705], [-8.12345, 41.134553]]])))
