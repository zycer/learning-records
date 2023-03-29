import time
import pickle
import random

if __name__ == '__main__':
    current_time = time.time()
    road_type = ["A", "B", "C", "D", "E"]
    for i in range(5):
        st_graph_data = {}
        for j in range(100):
            timestamp = current_time + j * 1800
            st_graph_data[timestamp] = {}
            for k in range(520):
                free_speed = random.randint(40, 140)
                length = random.uniform(100, 2000)
                speed = random.uniform(free_speed - 20, free_speed + 20)
                lanes = random.randint(1, 6)
                st_graph_data[timestamp][k] = (free_speed, length, lanes, speed, road_type[random.randint(0, len(road_type)-1)])
        with open(f"data/test_data_{i}.pickle", "wb") as f:
            pickle.dump(st_graph_data, f)
        print(f"data/test_data_{i}.pickle")
