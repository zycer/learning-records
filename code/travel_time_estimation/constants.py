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

ROAD_DATA_PATH = "data/road_data"
INTERSEC_DATA_PATH = "data/vertex_data"
ROAD_ATTR = ["link_id", "from_node_id", "to_node_id", "name", "length", "free_speed", "average_speed", "geometry"]
INTERSEC_ATTR = ["node_id", "name", "x_coord", "y_coord"]
