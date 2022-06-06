import json
import time
import math

import redis
from utils.constants import REDIS_INFO
from utils.db_manager import DBManager


def update_data():
    exists_num = len(exists)
    not_exists_num = len(not_exists)

    if not_exists:
        insert_sql = "INSERT INTO history_road_data(road_id,history) values "
        for road_id, history_data in not_exists.items():
            insert_sql += f"({road_id},'{json.dumps(history_data)}'),"

        insert_sql = insert_sql[:-1]
        db_handler.exec_sql(insert_sql)

    if exists:
        for road_id, history_data in exists.items():
            update_sql = f"UPDATE history_road_data SET history=concat_ws(';',history,'{json.dumps(history_data)}') where road_id={road_id}"
            db_handler.exec_sql(update_sql)

    db_handler.exec_sql(f"UPDATE finish_flag set num=num+{matched_num} where file_name='train.csv'")

    print(f"{int(time.time())}-->共{matched_num}条数据, 更新数据<{exists_num}>项，插入数据<{not_exists_num}>项")
    exists.clear()
    not_exists.clear()


if __name__ == '__main__':
    r = redis.Redis(**REDIS_INFO, decode_responses=True)
    db_handler = DBManager()

    known_roads = []
    result = db_handler.exec_sql("SELECT road_id FROM history_road_data")

    for res in result:
        known_roads.append(res[0])

    exists = {}
    not_exists = {}

    start_time = time.time()
    matched_num = 0

    while True:
        one_matched = r.rpop("matched_result")
        t = time.time()
        if one_matched:
            matched_num += 1
            matched = json.loads(one_matched)
            timestamp = matched["timestamp"]
            road_info = matched["road_info"]

            for idx, speeds in road_info.items():
                average_speed = round(sum(speeds) / len(speeds), 2)
                if idx not in exists.keys():
                    if idx in known_roads:
                        exists[idx] = {}
                        exists[idx][timestamp] = average_speed
                    else:
                        res = db_handler.exec_sql(f"SELECT road_id FROM history_road_data WHERE road_id={idx}")
                        if res:
                            exists[idx] = {}
                            exists[idx][timestamp] = average_speed
                            known_roads.append(idx)
                        else:
                            if idx in not_exists.keys():
                                not_exists[idx][timestamp] = average_speed
                            else:
                                not_exists[idx] = {timestamp: average_speed}
                else:
                    exists[idx][timestamp] = average_speed

        else:
            if exists or not_exists:
                update_data()
                matched_num = 0
            else:
                print("redis中无缓存，等待数据...")
                time.sleep(10)
                continue

        if time.time() - start_time > 10:
            update_data()
            matched_num = 0
            start_time = time.time()
