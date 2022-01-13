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
        self.segment_id_list = []
        self.segment_lines = []

    def data_pretreatment(self):
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

    def matched_segments(self):
        matched = []
        result = []
        self.data_pretreatment()
        for num, segment_line in enumerate(self.segment_lines):
            if len(segment_line) > 0:
                lines = np.concatenate(segment_line)
                tree = spatial.cKDTree(lines)
                lines_ix = tuple(itertools.chain.from_iterable(
                    [itertools.repeat(i, road) for i, road in enumerate(list(map(len, segment_line)))]
                ))
                distance, roads_idx = tree.query([self.trajectory[num]], k=self.k)
                matched.append(itemgetter(*roads_idx[0])(lines_ix))

            else:
                matched.append(())

        for num, lines in enumerate(matched):
            temp = []
            for line_index in lines:
                temp.append(self.segment_id_list[num][line_index])
            result.append(temp)
        print("原始匹配编号：", matched)
        self.plot_result()
        return result

    def plot_result(self):

        for i, segment in enumerate(self.segment_lines):
            for j, lines in enumerate(segment):
                plt.plot([lines[0][0], lines[1][0]], [lines[0][1], lines[1][1]],
                         label=f"{j}-{self.segment_id_list[i][j]}")

            plt.scatter(self.trajectory[i][0], self.trajectory[i][1])
            plt.legend(loc=0, ncol=2)
            plt.show()

            #     for line in lines:
            #         x_list.append(line[0])
            #         y_list.append(line[1])
            # print(x_list)
            # print(y_list)
            # print("~~~~~~~~~~")
            # plt.plot(x_list[i], y_list[i])
            # # plt.scatter(x_list, y_list)
            # # plt.legend(loc=0, ncol=2)
            # plt.show()
        # for j in range(len(self.segment_lines)):
        #     for line in self.segment_lines[j]:
        #         for num, value in enumerate(zip(line[0], line[1])):
        #             if num % 2 == 0:
        #                 x_list.append(value)
        #             else:
        #                 y_list.append(value)
        #
        #     point_x = []
        #     point_y = []
        #     for point in self.trajectory:
        #         point_x.append(point[0])
        #         point_y.append(point[1])
        #
        #     for i in range(len(x_list)):
        #         plt.plot(x_list[i], y_list[i], label=i)
        #
        #     plt.scatter(point_x, point_y)
        #     plt.legend(loc=0, ncol=2)
        #     plt.show()


if __name__ == "__main__":
    # points = [[7, 5], [27, 12]]
    # liness = [[[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]], [[18, 9], [20, 15]],
    #            [[12, 3.1], [17, 5.5]], [[13, 7.8], [15.5, 16]]],
    #           [[[0, 5], [3, 0.8]], [[4.5, 1.2], [9, 4.5]], [[11, 5.6], [4.8, 10.5]], [[18, 9], [20, 15]],
    #            [[12, 3.1], [17, 5.5]], [[13, 7.8], [15.5, 16]]]]
    # knn = KNN(points, liness)
    # knn.segment_id_list = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    # knn.matched_segments()
    # knn.plot_result()
    x = [[1.23, 3.56], [7.56, 8.66]]
    y = [[4.2, 5.44], [9.99, 6.66]]
    plt.plot(x, y)
    plt.show()

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
