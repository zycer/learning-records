import matplotlib.pyplot as plt
from scipy import spatial
from operator import itemgetter
import itertools
import numpy as np


class KNN:
    def __init__(self, trajectory, segments, k=2):
        self.trajectory = trajectory
        self.segments = segments
        self.k = k
        self.segment_list = np.concatenate(segments)
        self.tree = spatial.cKDTree(self.segment_list)
        self.segments_ix = tuple(itertools.chain.from_iterable(
            [itertools.repeat(i, road) for i, road in enumerate(list(map(len, self.segments)))]
        ))

    def matched_segments(self):
        matched_segment_idx = []
        distance, roads_idx = self.tree.query(self.trajectory, k=self.k)
        for road_id in roads_idx:
            matched_segment_idx.append(itemgetter(*road_id)(self.segments_ix))
        print("结果：")
        print(matched_segment_idx, distance)

    def plot_result(self):
        x_list = []
        y_list = []
        for line in self.segments:
            for num, value in enumerate(zip(line[0], line[1])):
                if num % 2 == 0:
                    x_list.append(value)
                else:
                    y_list.append(value)

        point_x = []
        point_y = []
        for point in self.trajectory:
            point_x.append(point[0])
            point_y.append(point[1])

        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i], label=i)

        plt.scatter(point_x, point_y)
        plt.legend(loc=0, ncol=2)
        plt.show()


if __name__ == "__main__":
    points = [[7, 5], [10, 8]]
    lines = [[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]], [[18, 9], [20, 15]],
             [[12, 3.1], [17, 5.5]], [[13, 7.8], [15.5, 16]]]
    knn = KNN(points, lines)
    knn.matched_segments()
    knn.plot_result()

#
# def knn(gps_trajectory, segments, k=2):
#     """
#     param: gps_point: 浮动车gps轨迹点列表
#     eg. [[7, 5], [10, 8]]
#     param: segments: 要匹配的路段
#     eg. [[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]], [[18, 9], [20, 15]]]
#     获取距离gps点最近的k个路段
#     """
#     # matched_segment_idx = []
#     # segments_ix = tuple(itertools.chain.from_iterable(
#     #     [itertools.repeat(i, road) for i, road in enumerate(list(map(len, segments)))]
#     # ))
#     # segment_list = np.concatenate(segments)
#     # ckd_tree = spatial.cKDTree(segment_list)
#     # dist, idx = ckd_tree.query(gps_trajectory, k=k)
#     # for road_id in idx:
#     #     matched_segment_idx.append(itemgetter(*road_id)(segments_ix))
#
#     print("结果：")
#     print(matched_segment_idx, dist)
#     x_list = []
#     y_list = []
#     for line in segments:
#         for num, value in enumerate(zip(line[0], line[1])):
#             if num % 2 == 0:
#                 x_list.append(value)
#             else:
#                 y_list.append(value)
#
#     point_x = []
#     point_y = []
#     for point in gps_trajectory:
#         point_x.append(point[0])
#         point_y.append(point[1])
#
#     for i in range(len(x_list)):
#         plt.plot(x_list[i], y_list[i], label=i)
#
#     plt.scatter(point_x, point_y)
#     plt.legend(loc=0, ncol=2)
#     plt.show()



