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
        print("匹配路段id：", result)
        self.plot_result()
        return result

    def plot_result(self):
        for i, segment in enumerate(self.segment_lines):
            plt.figure(figsize=(20, 20), dpi=80)
            for j, lines in enumerate(segment):
                plt.plot([lines[0][0], lines[1][0]], [lines[0][1], lines[1][1]],
                         label=f"{j}-{self.segment_id_list[i][j]}")

            plt.scatter(self.trajectory[i][0], self.trajectory[i][1])
            plt.legend(loc=0, ncol=2)
            plt.show()


if __name__ == "__main__":
    x = [[1.23, 3.56], [7.56, 8.66]]
    y = [[4.2, 5.44], [9.99, 6.66]]
    plt.plot(x, y)
    plt.show()
