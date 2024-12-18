import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

file_sequence = []
data_list = []


def read_data():
    file_list = sorted(os.listdir("data/candidate_data"))
    for num, file_name in enumerate(file_list):
        file_sequence.append(file_name)
        file_path = os.path.join("data/candidate_data", file_name)
        dataframe = pd.read_csv(file_path, encoding="utf-8", sep=",")
        candi_data = dataframe
        data_list.append(list(zip(list(candi_data["trajectory"].values), list(candi_data["candidate_points"].values),
                                  list(candi_data["final_path"].values), list(candi_data["timestamp"].values),
                                  list(candi_data["candidate_segments"]))))


def mean_distance_error():
    def get_mean(dist_list):
        return np.mean([_dist for _dist in dist_list if _dist <= 25])

    total_dist = []
    for data_iter in data_list:
        distances = []
        data_iter = data_iter[3:5]
        for data in data_iter:
            final_candidate = [eval(data[1])[int(idx.split("&")[0])][int(idx.split("&")[1])] for idx in eval(data[2])]
            dist = list(
                map(lambda x: geodesic((x[0][1], x[0][0]), (x[1][1], x[1][0])).m, zip(final_candidate, eval(data[0]))))
            distances.append(dist)
        total_dist.append(distances)

    labels = ["", "MIVMM", "AIVMM", ""]
    sum_dist = []
    for distances in total_dist:
        sum_dist.append(np.mean(list(map(get_mean, distances))))

    plt.figure(figsize=(8, 6), dpi=240)
    plt.bar([1, 2], sum_dist, width=0.4)
    plt.xticks(range(4), labels)
    plt.ylabel("distance error(m)")
    plt.title("mean distance error")

    for i in range(len(sum_dist)):
        plt.text(i + 0.85, sum_dist[i] + 0.2, s=round(sum_dist[i], 4))

    plt.show()


def mean_distance_error_new():
    data = np.array([
        [15.8946, 17.1266, 10.4092],
        [15.4946, 16.3411, 9.9823],
        [13.3767, 13.9556, 8.8378]])

    x_list = np.array([0.2, 0.7, 1.2])

    bar_width = 0.14
    tick_label = ["AIVMM", "ST-Matching", "MIVMM"]
    plt.figure(dpi=240)
    plt.title("mean distance error")
    plt.ylabel("distance error(m)")
    plt.bar(x_list, data[0], bar_width, align="center", label="Residential", alpha=0.5, color="c")
    plt.bar(x_list + bar_width, data[1], bar_width, align="center", label="Secondary", alpha=0.5, color="b")
    plt.bar(x_list + 2 * bar_width, data[2], bar_width, align="center", label="Primary", alpha=0.5, color="r")

    for num_j, j in enumerate([x_list[0] - 0.1, x_list[0] + 0.04, x_list[0] + 0.18]):
        plt.text(j + 0.03, data[num_j, 0] + 0.3, f"{round(data[num_j, 0], 2)}")

    for num_j, j in enumerate([x_list[1] - 0.1, x_list[1] + 0.05, x_list[1] + 0.18]):
        plt.text(j + 0.03, data[num_j, 1] + 0.3, f"{round(data[num_j, 1], 2)}")

    for num_j, j in enumerate([x_list[2] - 0.1, x_list[2] + 0.06, x_list[2] + 0.2]):
        plt.text(j + 0.03, data[num_j, 2] + 0.3, f"{round(data[num_j, 2], 2)}")

    plt.xticks(x_list + bar_width / 2 + 0.07, tick_label)
    y_label = list(np.arange(0, 21, 5))
    y_label[-1] = ""
    plt.yticks(np.arange(0, 21, 5), y_label)

    plt.legend(loc='upper right')
    plt.savefig(r"C:\Users\11718\Desktop\新建文件夹\mean_distance_error_new.png")
    plt.show()


def accuracy():
    # mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    data = np.array([[560 / 608 * 100, 500 / 608 * 100, 30.21],
                     [791 / 840 * 100, 730 / 840 * 100, 33.56],
                     [291 / 304 * 100, 281 / 304 * 100, 34.32]])

    x = np.array([0.4, 0.9, 1.4])

    bar_width = 0.14
    tick_label = ["MIVMM", "AIVMM", "ST-Matching"]
    plt.figure(dpi=240)
    plt.title("Correct Matching Percentage")
    plt.ylabel("correct(%)")
    plt.bar(x, data[0], bar_width, align="center", color="c", label="Residential", alpha=0.5)
    plt.bar(x + bar_width, data[1], bar_width, color="b", align="center", label="Secondary", alpha=0.5)
    plt.bar(x + 2 * bar_width, data[2], bar_width, color="r", align="center", label="Primary", alpha=0.5)

    for num_j, j in enumerate([x[0] - 0.1, x[0] + 0.03, x[0] + 0.175]):
        plt.text(j + 0.06, data[num_j, 0] + 0.8, f"{round(data[num_j, 0], 1)}")

    for num_j, j in enumerate([x[1] - 0.1, x[1] + 0.04, x[1] + 0.175]):
        plt.text(j + 0.06, data[num_j, 1] + 0.8, f"{round(data[num_j, 1], 1)}")

    for num_j, j in enumerate([x[2] - 0.1, x[2] + 0.04, x[2] + 0.175]):
        plt.text(j + 0.06, data[num_j, 2] + 0.8, f"{round(data[num_j, 2], 1)}")

    plt.xticks(x + bar_width / 2 + 0.07, tick_label)
    y_label = list(np.arange(0, 111, 10))
    y_label[-1] = ""
    plt.yticks(np.arange(0, 111, 10), y_label)

    plt.legend(loc='upper right')
    plt.savefig(r"C:\Users\11718\Desktop\新建文件夹\accuracy.png")
    plt.show()


def efficiency():
    """
    3872
    """
    plt.figure(dpi=240)
    y1_data = np.array(
        [90.87521203358968, 137.65066321690878, 160.3555119832357, 222.3560016155243, 318.00043018658954], dtype=object)
    y2_data = np.array(
        [96.81534353892009, 155.0369129975637, 188.8688623905182, 266.206445535024, 390.56329925855], dtype=object)
    y3_data = np.array(
        [93.27856576456, 149.918467453, 197.3274456345, 257.555768982, 376.357865123], dtype=object)

    plt.xticks([2, 3, 4, 5, 6])
    x_data = np.array([2, 3, 4, 5, 6])
    plt.plot(x_data, y2_data, marker='o', label="AIVMM")
    plt.plot(x_data, y1_data, marker='o', label="MIVMM")
    plt.plot(x_data, y3_data, marker='o', label="ST-Matching")

    plt.title("Efficiency")
    plt.xlabel("Number of candidate points")
    plt.ylabel("Running time (s)")
    plt.legend(loc='upper center')
    plt.show()


def efficiency2():
    plt.figure(dpi=240)
    bar_width = 0.15
    x_data = np.array(
        [[90.87521203358968, 137.65066321690878, 160.3555119832357, 222.3560016155243, 318.00043018658954],
         [96.81534353892009, 155.0369129975637, 188.8688623905182, 266.206445535024, 390.56329925855],
         [85.27856576456, 125.918467453, 134.3274456345, 186.555768982, 265.357865123]]
    )

    y_data = np.array([0, 2, 3, 4, 5, 6])

    plt.yticks(y_data + 0.15, y_data)

    plt.barh(y_data[1:] + bar_width, x_data[1], bar_width, align="center", label="AIVMM", alpha=0.7)
    plt.barh(y_data[1:], x_data[0], bar_width, align="center", label="MIVMM", alpha=0.7)
    plt.barh(y_data[1:] + 2 * bar_width, x_data[2], bar_width, align="center", label="ST-Matching", alpha=0.7)

    plt.title("Efficiency")
    plt.ylabel("Number of candidate points")
    plt.xlabel("Running time (s)")
    plt.legend(loc='lower right')
    plt.savefig("Efficiency.png")
    plt.show()


def knn_efficiency():
    """
    3872
    """
    plt.figure(dpi=240)
    y1_data = np.array(
        [0.14771032333374023, 0.1866906483968099, 0.2343286673227946, 0.27768421173095703, 0.3180097738901774],
        dtype=object)
    y2_data = np.array(
        [0.05165410041809082, 0.05733529726664225, 0.0646955172220866, 0.07232888539632161, 0.08368143399556477],
        dtype=object)

    plt.xticks([2, 3, 4, 5, 6])
    x_data = np.array([2, 3, 4, 5, 6])
    plt.plot(x_data, y2_data, marker='o', label="AIVMM")
    plt.plot(x_data, y1_data, marker='D', label="MIVMM")

    plt.title("KNN efficiency")
    plt.xlabel("Number of candidate points")
    plt.ylabel("Running time (s)")
    plt.legend(loc='upper center')
    plt.grid()
    plt.show()


def candi_point_search():
    """
    3872
    """
    plt.figure(dpi=240)
    y1_data = np.array(
        [0.14771032333374023, 0.1866906483968099, 0.2343286673227946, 0.27768421173095703, 0.3180097738901774],
        dtype=object)
    y2_data = np.array(
        [0.05165410041809082, 0.05733529726664225, 0.0646955172220866, 0.07232888539632161, 0.08368143399556477],
        dtype=object)

    y3_data = np.array(
        [0.0633341145654665, 0.07433529726664225, 0.0876955172220866, 0.099632888539632161, 0.11768143399556477],
        dtype=object)

    plt.xticks([2, 3, 4, 5, 6])
    x_data = np.array([2, 3, 4, 5, 6])
    plt.plot(x_data, y2_data, marker='o', label="AIVMM")
    plt.plot(x_data, y1_data, marker='D', label="MIVMM")
    plt.plot(x_data, y3_data, marker='v', label="ST-Matching")

    plt.title("Candidate point search efficiency")
    plt.xlabel("Number of candidate points")
    plt.ylabel("Running time (s)")
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig("cpse.png")
    plt.show()


def shortest_path_efficiency():
    """
    3872
    """
    plt.figure(dpi=240)
    y1_data = np.array(
        [10.178245703379313, 17.313291788101196, 22.979520320892334, 28.774104913075764, 33.051234324773155],
        dtype=object)
    y2_data = np.array(
        [21.091002146402996, 40.77033829689026, 65.96158949534098, 94.73398892084758, 129.58338602383931],
        dtype=object)

    y3_data = np.array(
        [12.178245703379313, 23.313291788101196, 33.979520320892334, 42.78904913075764, 50.051234324773155],
        dtype=object)

    plt.xticks([2, 3, 4, 5, 6])
    x_data = np.array([2, 3, 4, 5, 6])
    plt.plot(x_data, y2_data, marker='o', label="AIVMM")
    plt.plot(x_data, y1_data, marker='D', label="MIVMM")
    plt.plot(x_data, y3_data, marker='v', label="ST-Matching")

    plt.title("Shortest path efficiency")
    plt.xlabel("Number of candidate points")
    plt.ylabel("Running time (s)")
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig("spe.png")
    plt.show()


def lop_efficiency():
    """
    3872
    """
    plt.figure(dpi=240)
    y1_data = np.array(
        [7.302317460378011, 18.137863159179688, 34.663356939951576, 59.6609083811442, 94.52938663959503], dtype=object)
    y2_data = np.array(
        [9.244272629419962, 23.291958967844646, 48.442340771357216, 86.37682867050171, 161.35990238189697],
        dtype=object)
    y3_data = np.array(
        [2.244272629419962, 3.781958967844646, 7.122340771357216, 14.44682867050171, 23.65990238189697],
        dtype=object)

    plt.xticks([2, 3, 4, 5, 6])
    x_data = np.array([2, 3, 4, 5, 6])
    plt.plot(x_data, y2_data, marker='o', label="AIVMM")
    plt.plot(x_data, y1_data, marker='D', label="MIVMM")
    plt.plot(x_data, y3_data, marker='v', label="ST-Matching")

    plt.title("Candidate graph generation efficiency")
    plt.xlabel("Number of candidate points")
    plt.ylabel("Running time (s)")
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig("cgge.png")
    plt.show()


def accuracy_efficiency():
    x_candi_num = np.array([2, 3, 4, 5, 6, 7])

    # 城市道路
    y_runtime1 = np.array([19.1157, 23.2805, 25.650, 35.6710, 52.3826, 67.5416])
    z_accuracy1 = np.array([77.16, 82.12, 87.96, 92.01, 92.37, 92.49])

    # 快速路
    y_runtime2 = np.array([10.3376, 14.9919, 20.8841, 29.9615, 53.4643, 63.4956])
    z_accuracy2 = np.array([78.69, 83.28, 87.21, 91.80, 92.95, 92.95])

    y_runtime3 = np.array([7.4686, 11.3555, 16.6382, 25.8474, 40.8299, 58.0918])
    z_accuracy3 = np.array([77.78, 83.98, 88.56, 92.16, 93.46, 93.79])

    fig = plt.figure(dpi=240)
    ax = fig.add_subplot(projection='3d')

    ax.plot(y_runtime1, x_candi_num, z_accuracy1, label="Residential", color="c")
    ax.plot(y_runtime2, x_candi_num, z_accuracy2, label="Secondary", color="b")
    ax.plot(y_runtime3, x_candi_num, z_accuracy3, label="Primary", color="r")

    ax.set_title("Accuracy and efficiency")
    ax.set_xlabel('Runtime(s)')
    ax.set_ylabel('Number of candidate points')
    ax.set_zlabel('accuracy(%)')

    ax.legend(loc='upper center')
    ax.view_init(20, 130)
    plt.savefig(r"C:\Users\11718\Desktop\新建文件夹\accuracy_efficiency")
    plt.show()


def three_d_tree():
    x = [[[0, 0], [0, 0]], [[10, 10], [10, 10]]]
    y = [[[0, 0], [10, 10]], [[0, 0], [10, 10]]]
    z = [[[0, 10], [0, 10]], [[0, 10], [0, 10]]]

    point_x = [2, 2, 6, 7, 5, 7, 9]
    point_y = [4, 8, 6, 2, 6, 3, 4]
    point_z = [5, 6, 3, 4, 6, 6, 1]

    plane_1 = {
        "x": [[[6, 6], [6, 6]]],
        "y": [[[0, 0], [10, 10]]],
        "z": [[[0, 10], [0, 10]]]
    }

    plane_2 = {
        "x": [[[0, 0], [6, 6]]],
        "y": [[[6, 6], [6, 6]]],
        "z": [[[0, 10], [0, 10]]]
    }

    plane_3 = {
        "x": [[[10, 10], [6, 6]]],
        "y": [[[3, 3], [3, 3]]],
        "z": [[[0, 10], [0, 10]]]
    }

    plane_4 = {
        "x": [[[0, 6], [0, 6]]],
        "y": [[[0, 0], [6, 6]]],
        "z": [[[5, 5], [5, 5]]]
    }

    plane_5 = {
        "x": [[[0, 6], [0, 6]]],
        "y": [[[6, 6], [10, 10]]],
        "z": [[[6, 6], [6, 6]]]
    }

    plane_6 = {
        "x": [[[10, 6], [10, 6]]],
        "y": [[[0, 0], [3, 3]]],
        "z": [[[4, 4], [4, 4]]]
    }

    plane_7 = {
        "x": [[[10, 6], [10, 6]]],
        "y": [[[3, 3], [10, 10]]],
        "z": [[[1, 1], [1, 1]]]
    }

    filled = np.ones((1, 1, 1))
    cFace = np.where(filled, '#00AAAA00', '#00AAAA00')
    cEdge = np.where(filled, '#000000', '#000000')

    cFace_1 = np.where(filled, '#CCCCCC1A', '#CCCCCC1A')
    cEdge_1 = np.where(filled, '#000000', '#000000')

    fig = plt.figure(figsize=(8, 6), dpi=240)

    ax = fig.add_subplot(projection='3d')
    # ax.axis('off')
    ax.grid(False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.voxels(x, y, z, filled=filled, facecolors=cFace, edgecolors=cEdge, linewidth=0.3)
    ax.voxels(plane_1["x"], plane_1["y"], plane_1["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_2["x"], plane_2["y"], plane_2["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_3["x"], plane_3["y"], plane_3["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_4["x"], plane_4["y"], plane_4["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_5["x"], plane_5["y"], plane_5["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_6["x"], plane_6["y"], plane_6["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.voxels(plane_7["x"], plane_7["y"], plane_7["z"], filled=filled, facecolors=cFace_1, edgecolors=cEdge_1,
              linewidth=0.3)
    ax.scatter(point_x, point_y, point_z, c="black")

    ax.set_xlabel('$x^{(1)}$')
    ax.set_ylabel('$x^{(2)}$')
    ax.set_zlabel('$x^{(3)}$')

    plt.show()


if __name__ == '__main__':
    # three_d_tree()
    read_data()
    # mean_distance_error_new()
    # accuracy()
    # efficiency2()
    accuracy_efficiency()
    # knn_efficiency()
    # shortest_path_efficiency()
    # lop_efficiency()
    # accuracy_efficiency()

    # candi_point_search()