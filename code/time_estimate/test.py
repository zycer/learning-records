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
    for file_name in os.listdir("data/gps_trajectory"):
        df = pd.read_csv(os.path.join("data/gps_trajectory", file_name))

