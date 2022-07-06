import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import matplotlib as mpl


file_sequence = []
data_list = []


def read_data():
    for num, file_name in enumerate(os.listdir("data/candidate_data")):
        file_sequence.append(file_name)
        file_path = os.path.join("data/candidate_data", file_name)
        dataframe = pd.read_csv(file_path, encoding="utf-8", sep=",")
        candi_data = dataframe
        data_list.append(list(zip(list(candi_data["trajectory"].values), list(candi_data["candidate_points"].values),
                                  list(candi_data["final_path"].values), list(candi_data["timestamp"].values),
                                  list(candi_data["candidate_segments"]))))


def mean_distance_error():
    total_dist = []
    for data_iter in data_list:
        distances = []
        data_iter = data_iter[1:2]
        for data in data_iter:
            final_candidate = [eval(data[1])[int(idx.split("&")[0])][int(idx.split("&")[1])] for idx in eval(data[2])]
            dist = list(
                map(lambda x: geodesic((x[0][1], x[0][0]), (x[1][1], x[1][0])).m, zip(final_candidate, eval(data[0]))))
            distances.append(dist)
        total_dist.append(distances)

    labels = ["", "MIVMM", "AIVMM", ""]
    sum_dist = []
    for distances in total_dist:
        sum_dist.append(np.mean(list(map(lambda x: np.mean(x), distances))))

    plt.figure(figsize=(8, 6), dpi=240)
    plt.bar([1, 2], sum_dist, width=0.4)
    plt.xticks(range(4), labels)
    plt.ylabel("distance error(m)")
    plt.title("mean distance error")

    for i in range(len(sum_dist)):
        plt.text(i + 0.85, sum_dist[i] + 0.2, s=round(sum_dist[i], 4))

    plt.show()


def accuracy():
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    data = np.array([[560 / 608 * 100, 500 / 608 * 100, 0],
                     [791 / 840 * 100, 730 / 840 * 100, 0],
                     [291 / 304 * 100, 280 / 304 * 100, 0]])

    branch = [560/608*100, 500/608*100, 0]
    expressway = [791/840*100, 730/840*100, 0]
    highway = [291/304*100, 280/304*100, 0]
    x = np.array([0.2, 0.7, 1])
    print(x)

    bar_width = 0.14
    tick_label = ["MIVMM", "AIVMM"]
    plt.figure(dpi=240)
    plt.title("Correct Matching Percentage")
    plt.ylabel("correct(%)")
    plt.bar(x, data[0], bar_width, align="center", color="c", label="Branch", alpha=0.5)
    plt.bar(x + bar_width, data[1], bar_width, color="b", align="center", label="Expressway", alpha=0.5)
    plt.bar(x + 2 * bar_width, data[2], bar_width, color="r", align="center", label="Highway", alpha=0.5)

    for num_j, j in enumerate([x[0]-0.1, x[0]+0.04, x[0]+0.17]):
        plt.text(j+0.06, data[num_j, 0]+0.8, f"{round(data[num_j, 0], 2)}")

    for num_j, j in enumerate([x[1]-0.1, x[1]+0.05, x[1]+0.18]):
        plt.text(j+0.06, data[num_j, 1]+0.8, f"{round(data[num_j, 1], 2)}")

    plt.xticks(x[:-1] + bar_width / 2 + 0.07, tick_label)
    y_label = list(np.arange(0, 111, 10))
    y_label[-1] = ""
    plt.yticks(np.arange(0, 111, 10), y_label)

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    read_data()
    # mean_distance_error()
    accuracy()
