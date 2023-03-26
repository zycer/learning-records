ROAD_MAX_SPEED = {
    "steps": 5,
    "living_street": 10,
    "secondary": 60,
    "motorway_link": 120,
    "primary": 80,
    "pedestrian": 50,
    "raceway": 70,
    "road": 60,
    "primary_link": 80,
    "residential": 30,
    "trunk": 80,
    "trunk_link": 80,
    "service": 50,
    "track": 60,
    "tertiary_link": 60,
    "tertiary": 60,
    "unclassified": 60,
    "motorway": 120,
    "secondary_link": 60,
    "cycleway": 20,
    "path": 20,
    "footway": 10,
    "other": 50
}

ROAD_DATA_PATH = "../../map_matching/data/road_network/other/road_data"
INTERSEC_DATA_PATH = "../../map_matching/data/road_network/other/vertex_data"
GRAPH_DATA = "../../map_matching/data/graph_data"
ROAD_ATTR = ["link_id", "from_node_id", "to_node_id", "name", "length", "lanes",
             "free_speed", "average_speed", "geometry", "type_name"]
INTERSEC_ATTR = ["intersection_id", "name", "x_coord", "y_coord"]

REDIS_INFO = {
            "host": "127.0.0.1",
            "password": "",
            "port": 6379,
            "db": 0
        }
