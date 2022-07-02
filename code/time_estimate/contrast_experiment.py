import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

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
        for data in data_iter:
            final_candidate = [eval(data[1])[int(idx.split("&")[0])][int(idx.split("&")[1])] for idx in eval(data[2])]
            dist = list(
                map(lambda x: geodesic((x[0][1], x[0][0]), (x[1][1], x[1][0])).m, zip(final_candidate, eval(data[0]))))
            distances.append(dist)
        total_dist.append(distances)

    labels = ["", "ME", "AIVMM", ""]
    sum_dist = []
    for distances in total_dist:
        sum_dist.append(np.mean(list(map(lambda x: np.mean(x), distances))))

    plt.bar([1, 2], sum_dist, width=0.4)
    plt.xticks(range(4), labels)
    plt.ylabel("distance error(m)")
    plt.title("mean distance error")

    for i in range(len(sum_dist)):
        plt.text(i + 0.85, sum_dist[i] + 0.2, s=round(sum_dist[i], 4))

    plt.show()


def accuracy():
    points_num = sum(map(lambda x: len(eval(x[0])), data_list[0]))
    print(points_num)
    aivmm_num = 764
    me_num = 839
    x_labels = ["", "me", "other", ""]
    y_data = [me_num / points_num * 100, aivmm_num / points_num * 100]
    plt.bar([1, 2], y_data, width=0.5)
    plt.title("Correct Matching Percentage")
    plt.ylabel("correct(%)")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(np.arange(0, 101, 10))

    for i in range(len(y_data)):
        plt.text(i + 0.85, y_data[i] + 1, f"{round(y_data[i], 2)}%")

    plt.show()


if __name__ == '__main__':
    read_data()
    mean_distance_error()
    accuracy()
