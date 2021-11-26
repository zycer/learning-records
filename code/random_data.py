import random
import numpy as np


def func1(x):
    return 6 * x


def func2(x):
    return 5.8 * x + 1.2


def func3(x):
    return 6.15 * x - 2.33


def disturbance(value, seed, positive=True):
    if positive:
        if value - seed < 0:
            return 0
        else:
            return value - seed
    else:
        return value - seed


if __name__ == "__main__":
    func_set = (func1, func2, func3)

    with open("data/data.csv", "w") as f:
        f.write("soc,time,distance\n")
        data = set()
        for x in np.arange(0, 100.0, 1):
            index = random.randint(0, len(func_set) - 1)
            ran = random.random()
            if ran > 0.8:
                time_seed = random.uniform(-10, 12)
                distance_seed = random.uniform(-60, 60)
            else:
                time_seed = random.uniform(-30, 20)
                distance_seed = random.uniform(-60, 60)

            soc = round(x, 2)
            time = round(disturbance(x, time_seed))
            distance = round(disturbance(func_set[index](x), distance_seed) - 0.23 * time, 2)
            data.add("%s,%s,%s\n" % (soc, time, distance if distance > 0 else 0))

        for info in data:
            f.write(info)
