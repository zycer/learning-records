import random


new_data = []
with open("data/road_graph.csv", encoding="utf-8") as f:
    data_list = f.readlines()
    for num, segment in enumerate(data_list):
        if num == 0:
            new_data.append(segment)
            continue
        mileage = random.uniform(2, 20)
        new_segment = segment.strip()
        new_segment += f",{mileage}\n"
        new_data.append(new_segment)


with open("data/road_graph/new_road_graph.csv", "w", encoding="utf-8") as ff:
    ff.writelines(new_data)

