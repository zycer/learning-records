import os
import osm2gmns as og
import xml.etree.ElementTree as ET

speed_limit = {
    "steps": 5,
    "living_street": "10",
    "secondary": "60",
    "motorway_link": "120",
    "primary": "80",
    "pedestrian": "50",
    "raceway": "70",
    "road": "60",
    "primary_link": "80",
    "residential": "30",
    "trunk": "80",
    "trunk_link": "80",
    "service": "50",
    "track": "60",
    "tertiary_link": "60",
    "tertiary": "60",
    "unclassified": "60",
    "motorway": "120",
    "secondary_link": "60",
    "cycleway": "20",
    "path": "20",
    "footway": "10",
    "other": "50"
}


def trans_osm2graph(osm_path, out_path):
    net = og.getNetFromFile(osm_path, link_types=('motorway', 'trunk', 'primary', 'secondary', 'tertiary'),
                            default_lanes=True, default_speed=True, default_capacity=True)
    og.consolidateComplexIntersections(net, auto_identify=True)
    og.outputNetToCSV(net, output_folder=out_path)


def format_link_data(link_data, max_speed_list, out_path):
    titles = link_data[0].strip() + ",max_speed,average_speed\n"
    new_link_data = []
    points = []

    with open(os.path.join(out_path, "new_link.csv"), "w", encoding="utf-8") as f:
        f.write(titles)

    for i, link in enumerate(link_data[1:]):
        link_attr = []
        link_pre, points_str, link_behind = link.split('"')
        for point_str in points_str[12: -1].split(","):
            points.append([float(i) for i in point_str.strip().split(" ")])

        link_attr.extend([i.strip() for i in link_pre.split(",")][:-1])

        geometry_str = "|["
        for j in range(len(points)):
            geometry_str += f"{points[j]},"
        geometry_str += f"{points[-1]}]|"

        link_attr.append(geometry_str)
        link_attr.extend([i.strip() for i in link_behind.split(",")][1:])

        link_attr.append(max_speed_list[i])
        link_attr.append(max_speed_list[i])

        link_attr = [str(i) for i in link_attr]
        new_link_data.append(",".join(link_attr))

        if len(new_link_data) > 100:
            with open(os.path.join(out_path, "new_link.csv"), "a+", encoding="utf-8") as f:
                f.write("\n".join(new_link_data))
                f.write("\n")
            print("写入link，数据量100")
            new_link_data.clear()

    with open(os.path.join(out_path, "new_link.csv"), "a+", encoding="utf-8") as f:
        f.write("\n".join(new_link_data))


def format_vertex_data(node_file, out_path):
    with open(node_file, "r", encoding="utf-8") as f:
        titles = "vertex_id,longitude,latitude\n"
        with open(os.path.join(out_path, "new_node.csv"), "w", encoding="utf-8") as ff:
            ff.write(titles)
        vertex_data = []
        node_data = [link.strip() for link in f.readlines()]
        for node in node_data[1:]:
            node_attr = node.split(",")
            node_id = node_attr[1]
            longitude = node_attr[9]
            latitude = node_attr[10]
            vertex_data.append(f"{node_id},{longitude},{latitude}")

            if len(vertex_data) > 100:
                with open(os.path.join(out_path, "new_node.csv"), "a+", encoding="utf-8") as ff:
                    ff.write("\n".join(vertex_data))
                    ff.write("\n")
                print("写入vertex，数据量100")
                vertex_data.clear()

        with open(os.path.join(out_path, "new_node.csv"), "a+", encoding="utf-8") as ff:
            ff.write("\n".join(vertex_data))
            ff.write("\n")


def check_graph_data(graph_path):
    link_file = os.path.join(graph_path, "link.csv")
    node_file = os.path.join(graph_path, "node.csv")

    with open(link_file, "r", encoding="utf-8") as f:
        # tree = ET.parse(os.path.join(graph_path, "shenzhen.osm"))
        # root = tree.getroot()
        link_data = [link.strip() for link in f.readlines()]
        max_speed_list = []

        for link in link_data[1:]:
            link_attr = [i.strip() for i in link.split(",")]
            # way_id = link_attr[2]
            # way = root.find(f"./way[@id='{way_id}']")
            # if way is not None:
            #     tag = way.find("./tag[@maxspeed]")
            #     if tag is not None:
            #         max_speed = tag.attrib["v"]
            #         print("此道路存在限速")
            #     else:
            #         tag = way.find("./tag[@highway]")
            #         if tag is not None:
            #             way_type = tag.attrib['v']
            #             max_speed = speed_limit[way_type] if way_type in speed_limit.keys() else speed_limit["other"]
            #         else:
            #             max_speed = speed_limit["other"]
            # else:
            #     max_speed = -1
            if link_attr[10] in speed_limit.keys():
                max_speed = speed_limit[link_attr[10]]
            else:
                max_speed = speed_limit["other"]

            max_speed_list.append(max_speed)
        format_link_data(link_data, max_speed_list, "G:\data")
        format_vertex_data(node_file, "G:\data")


if __name__ == "__main__":
    trans_osm2graph("data/osm_data/Porto.osm", "data/osm_data")
    # check_graph_data("data/osm_data")
