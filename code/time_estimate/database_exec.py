import json
import time
import math

import redis
from utils.constants import REDIS_INFO
from utils.db_manager import DBManager


if __name__ == '__main__':
    result = []
    last_time = time.time()
    r = redis.Redis(**REDIS_INFO, decode_responses=True)
    db_handler = DBManager()

    while True:
        one_data = r.rpop("matched_result")
        exist_data = []
        not_exist_idx = []

        time.sleep(1)
        if one_data:
            result.append(one_data)

        if time.time()-last_time >= 1 and len(result) > 0:
            sql = "INSERT"
            for res in result:
                res = json.loads(res)
                timestamp = res["timestamp"]
                road_info = res["road_info"]

                road_ids = list(road_info.keys())

                query_sql = "SELECT road_id,history from history_road_data where "
                for idx in road_ids:
                    query_sql += f"road_id={idx} or "
                query_sql = query_sql[:-3]

                query_data = db_handler.exec_sql(query_sql)

                exist_ids = []
                for qd in query_data[0]:
                    exist_ids.append(qd[0])
                    exist_data.append((qd[0], json.loads(qd[1])))
                not_exist_idx.extend(list(set(road_ids) - set(exist_ids)))
                time.sleep(3)

                for road_id, speed_list in road_info.items():
                    average_speed = round(sum(speed_list) / len(speed_list), 2)
