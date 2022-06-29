import os
import folium
import webbrowser
import pandas as pd
import random


def save_map(map_obj, map_name):
    map_name = map_name.replace(".csv", ".html")
    map_file_path = os.path.join("data/temp", map_name)
    map_obj.save(map_file_path)
    search_text = "cdn.jsdelivr.net"
    replace_text = "fastly.jsdelivr.net"

    with open(map_file_path, 'r', encoding='UTF-8') as file:
        data = file.read()
        data = data.replace(search_text, replace_text)

    with open(map_file_path, 'w', encoding='UTF-8') as file:
        file.write(data)

    webbrowser.open(map_file_path, new=True)


def create_map(candi_file, start, end):
    tiles = {
        0: "OpenStreetMap",
        1: "Stamen Terrain",
        2: "Stamen Toner",
        3: "Mapbox Bright",
        4: "Mapbox Control Room"
    }
    radius = 4
    porto_map = folium.Map([41.141412, -8.618643], tiles=tiles[1], zoom_start=16)
    dataframe = pd.read_csv(candi_file, encoding="utf-8", sep=",")
    candi_data = dataframe.iloc[start:end]
    data_iter = zip(list(candi_data["trajectory"].values), list(candi_data["candidate_points"].values),
                    list(candi_data["final_path"].values), list(candi_data["timestamp"].values),
                    list(candi_data["candidate_segments"]))

    for index_1, data in enumerate(data_iter):
        color_list = []
        for index_2, point in enumerate(eval(data[0])):
            if first_flag:
                current_color = "#%06x" % random.randint(0, 0xFFFFFF)
                color_list.append(current_color)
            else:
                current_color = colors[index_1][index_2]

            folium.Circle(radius=radius, location=(point[1], point[0]),
                          popup=f"{str(data[3])}-{index_2}(采样点)", color=current_color, fill=True, fill_opacity=0.0).add_to(
                porto_map)

        final_candidate = [eval(data[1])[int(idx.split("&")[0])][int(idx.split("&")[1])] for idx in eval(data[2])]

        for idx, candi_point in enumerate(final_candidate):
            if first_flag:
                color = color_list[idx]
            else:
                color = colors[index_1][idx]
            folium.Circle(radius=radius, location=(candi_point[1], candi_point[0]),
                          popup=f"{str(data[3])}-{idx}(匹配点)", color=color, fill=True, fill_opacity=0.7).add_to(porto_map)

        final_candi_segments = [eval(data[4])[int(idx.split("&")[0])][int(idx.split("&")[1])] for idx in eval(data[2])]
        for idx, segment in enumerate(final_candi_segments):
            if first_flag:
                color = color_list[idx]
            else:
                color = colors[index_1][idx]
            point_a = segment[0]
            point_b = segment[1]
            folium.PolyLine(locations=[[point_a[1], point_a[0]],
                                       [point_b[1], point_b[0]]], color=color, weight=5).add_to(porto_map)

        # for idx, segments in enumerate(eval(data[4])):
        #     color = color_list[idx] if first_flag else colors[index_1][idx]
        #     weight = 1
        #     for seg in segments:
        #         point_a = seg[0]
        #         point_b = seg[1]
        #         folium.PolyLine(locations=[[point_a[1], point_a[0]],
        #                                    [point_b[1], point_b[0]]], color=color, weight=weight,
        #                         line_cap="square").add_to(porto_map)
        #         weight += 1

        # for idx, points in enumerate(eval(data[1])):
        #     color = color_list[idx] if first_flag else colors[index_1][idx]
        #     for point in points:
        #         folium.Circle(location=(point[1], point[0]), popup=f"{str(data[3])}-{idx}(其他候选点)", color=color, radius=2,
        #                       fill=True, fill_opacity=0.3).add_to(porto_map)

        colors.append(color_list)
    return porto_map


if __name__ == '__main__':
    start = 6
    end = start + 1
    first_flag = True
    colors = []
    for num, file_name in enumerate(os.listdir("data/candidate_data")):
        file_path = os.path.join("data/candidate_data", file_name)

        if num > 0:
            first_flag = False

        porto_map = create_map(file_path, start, end)
        save_map(porto_map, file_name)
