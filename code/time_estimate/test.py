import json

import pandas as pd
import os


def statistic_rows_num():
    total = 0
    for file_name in os.listdir("data/gps_trajectory"):
        df = pd.read_csv(os.path.join("data/gps_trajectory", file_name))
        print(df.shape)
        total += df.shape[0]

    print(total)


def statistic_points_num():
    total_number = 0
    for file_name in os.listdir("data/gps_trajectory"):
        df = pd.read_csv(os.path.join("data/gps_trajectory", file_name))
        for tra in map(json.loads, df["POLYLINE"].values):
            total_number += len(tra)

        print("Operation to complete, %s" % file_name)

    print(total_number)


statistic_points_num()
