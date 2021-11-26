import random
import numpy as np


def func1(x):
    return 3.15 * x + 3.11


def func2(x):
    return 3.01 * x + 2.99


def func3(x):
    return 2.89 * x + 2.33


if __name__ == "__main__":
    func_set = (func1, func2, func3)

    with open("data/test.csv", "w") as f:
        f.write("a,b\n")
        for i in np.arange(1.0, 10.0, 0.1):
            index = random.randint(0, len(func_set)-1)
            ran = random.random() * 6
            f.write("%s,%s\n" % (round(i, 2), round(func_set[index](i) + ran, 2)))

