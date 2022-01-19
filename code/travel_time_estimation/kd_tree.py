import math
import time

import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import itertools
import numpy as np


class KNN:
    def __init__(self, trajectory, segments, neighbor_num=2):
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
        self.segments = segments
        self.neighbor_num = neighbor_num
        self.segment_lines = []
        self.segment_id_list = []

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

    def matched_segments(self, is_plot=True):
        """
        匹配k个邻居
        param: is_plot: 是否画图显示
        matched: [{(road_id, distance), (road_id, distance),...}, {(road_id, distance), (road_id, distance),...},...]
        """
        start_time = time.time()
        matched = []
        self.data_pretreatment()
        for num, segment_line in enumerate(self.segment_lines):
            last_matched = set()
            k = self.neighbor_num
            lines = np.concatenate(segment_line)
            tree = spatial.cKDTree(lines)
            lines_ix = tuple(itertools.chain.from_iterable(
                [itertools.repeat(i, road) for i, road in enumerate(list(map(len, segment_line)))]
            ))

            while len(segment_line) >= k:
                distance, roads_idx = tree.query([self.trajectory[num]], k=k)

                match_set = set()
                for index, segment_id in enumerate(itemgetter(*roads_idx[0])(lines_ix)):
                    temp = [segment[0] for segment in match_set]
                    if self.segment_id_list[num][segment_id] not in temp:
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
        return matched

    def plot_result(self):
        for i, segment in enumerate(self.segment_lines):
            plt.figure(figsize=(20, 20), dpi=80)
            for j, lines in enumerate(segment):
                plt.plot([lines[0][0], lines[1][0]], [lines[0][1], lines[1][1]],
                         label=f"{j}-{self.segment_id_list[i][j]}")

            plt.scatter(self.trajectory[i][0], self.trajectory[i][1])
            plt.legend(loc=0, ncol=2)
            plt.show()
