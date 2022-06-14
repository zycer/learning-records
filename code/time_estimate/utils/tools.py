import MySQLdb
import pandas as pd
import numpy as np

from constants import ROAD_MAX_SPEED as RMS


def get_road_data(road_path):
    def parse_road_nodes(geometry):
        nodes = []
        for points_str in geometry.split(","):
            longitude, latitude = points_str.strip().split(" ")
            nodes.append([float(longitude), float(latitude)])
        return nodes

    # db_handler = DBManager()
    # res = db_handler.exec_sql(f"SELECT road_id, average_speed FROM history_road_data")
    known_roads = {}
    # for data in res:
    #     known_roads[data[0]] = data[1]

    road_data = pd.read_csv(road_path, encoding="utf-8", sep=",")
    road_names = road_data["name"].values
    road_ids = road_data["link_id"].values
    from_vertexes = road_data["from_node_id"].values
    to_vertexes = road_data["to_node_id"].values
    geometries = list(map(parse_road_nodes, np.array(list(map(lambda x: x[12:-1], road_data["geometry"].values)))))
    mileages = road_data["length"].values
    average_speed = [known_roads[idx] if idx in known_roads.keys() else 0 for idx in road_ids]
    free_speed = [RMS.get(road_data["link_type_name"][i], RMS["other"]) if np.isnan(f_speed) else f_speed for i, f_speed
                  in enumerate(road_data["free_speed"].values)]

    return zip(road_ids, from_vertexes, to_vertexes, road_names, mileages, free_speed, average_speed, geometries)


def get_intersection_data(interaction_path):
    intersection_data = pd.read_csv(interaction_path, encoding="utf-8", sep=",")
    longitudes = intersection_data["x_coord"].values
    latitudes = intersection_data["y_coord"].values
    intersection_names = intersection_data["name"].values
    intersection_ids = intersection_data["node_id"].values
    return zip(intersection_ids, intersection_names, longitudes, latitudes)


def get_graph_data(graph_path):
    graph_data = pd.read_csv(graph_path, encoding="utf-8", sep=",")
    from_ids = graph_data["from_id"].values
    to_ids = graph_data["to_id"].values
    node_ids = graph_data["node_id"].values
    return zip(from_ids, to_ids, node_ids)


class DBManager:
    def __init__(self):
        self.conn = MySQLdb.connect(
            host='localhost',
            port=3306,
            user='root',
            passwd='123456',
            db='ev_estimate'
        )
        self.cur = self.conn.cursor()

    def exec_sql(self, sql):
        self.cur.execute(sql)
        if "SELECT" not in sql:
            self.conn.commit()
        return self.cur.fetchall()

    def close(self):
        self.cur.close()
        self.conn.close()
