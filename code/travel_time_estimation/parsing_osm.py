import os
import osm2gmns as og
import xml.etree.ElementTree as ET

speed_limit = {
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


def trans_osm2graph(osm_path, out_path):
    net = og.getNetFromFile(osm_path, link_types=('motorway', 'trunk', 'primary', 'secondary', 'tertiary'))
    og.outputNetToCSV(net, output_folder=out_path)


def format_link_data(link_data, max_speed_list):
    titles = link_data[0].strip() + ",max_speed"
    for link in link_data[1:]:
        link_attr = link.split(",")
        geometry = link_attr[12][13: -2]
        print(geometry)



def check_graph_data(graph_path):
    link_file = os.path.join(graph_path, "link.csv")
    node_file = os.path.join(graph_path, "node.csv")

    with open(link_file, "r", encoding="utf-8") as f:
        tree = ET.parse("data/osm_data/shenzhen.osm")
        root = tree.getroot()
        link_data = [link.strip() for link in f.readlines()]
        max_speed_list = []

        for link in link_data[1:]:
            way_id = link[2]
            way = root.find(f"./way[@id={way_id}]")
            if way is not None:
                tag = way.find("./tag[@maxspeed]")
                if tag is not None:
                    max_speed = int(tag.attrib["v"])
                else:
                    tag = way.find("./tag[@highway]")
                    if tag is not None:
                        way_type = tag.attrib['v']
                        max_speed = speed_limit[way_type] if way_type in speed_limit.keys() else speed_limit["other"]
                    else:
                        max_speed = speed_limit["other"]
            else:
                raise KeyError

            max_speed_list.append(max_speed)

        format_link_data(link_data, max_speed_list)


# check_graph_data("data/osm_data")

a = '"LINESTRING (114.1079143 22.5415774, 114.1072354 22.5414963)"'

geometry = a[12][13: -2]
road_nodes = [[j[0], j[1]]for i in geometry.split(",") for j in i.split(" ")]
print(road_nodes)
