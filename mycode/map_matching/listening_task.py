import json
import os
import time

import pandas as pd
import redis

from utils.db_manager import DBManager


def listening_task():
    redis_info = {
        "host": "127.0.0.1",
        "password": "",
        "port": 6379,
        "db": 0
    }

    r = redis.Redis(**redis_info, decode_responses=True)
    r.delete("trajectory")
    trajectory_data = {}
    file_path = "data/gps_trajectory"
    result = DBManager().exec_sql("select file_name,num from finish_flag")
    finish_dict = {}

    for res in result:
        finish_dict[res[0]] = res[1]

    for file_name in os.listdir(file_path):
        if file_name not in finish_dict.keys():
            DBManager().exec_sql(f"insert into finish_flag(file_name,num) value ('{file_name}',0)")
            finish_dict[file_name] = 0
        tra_data = pd.read_csv(os.path.join(file_path, file_name), sep=",", encoding="utf-8", iterator=True)
        trajectory_data[file_name] = tra_data

    for name, tra_iter in trajectory_data.items():
        tra_iter.get_chunk(finish_dict[name])
        try:
            while True:
                if r.llen("trajectory") > 1000:
                    time.sleep(10)
                else:
                    tra_one = tra_iter.get_chunk(1)
                    polyline = json.loads(tra_one["POLYLINE"].values[0])
                    timestamp = tra_one["TIMESTAMP"].values[0].item()
                    r.rpush("trajectory", json.dumps({"polyline": polyline, "timestamp": timestamp, "file_name": name}))
        except StopIteration:
            break
