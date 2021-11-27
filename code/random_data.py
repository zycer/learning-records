import random
import numpy as np


class EV:

    attribute = [
        "soc",
        "time",
        "distance"
    ]

    def __init__(self, name, k, b, t_discount):
        self.name = name
        self.k = k
        self.b = b
        self.file_path = f"./data/{self.name}.csv"
        self.t_discount = t_discount
        self.data = set()

    def linear_equation(self, x, time):
        self.k = random.uniform(-0.2, 0.2) + self.k
        self.b = random.uniform(-1.8, 1.8) + self.b
        return self.k * x + self.b - self.t_discount * time

    @classmethod
    def disturbance(cls, value, positive=True):
        dis_range = 10
        ran = random.random()
        dis_num = 2 * dis_range if ran > 0.8 else 3 * dis_range
        value += random.uniform(-dis_num, dis_num)
        if positive:
            return value if value > 0 else 0
        else:
            return value

    def generate_ev_data(self):
        with open(self.file_path, "w") as f:
            f.write(','.join(self.attribute) + "\n")
            for i in np.arange(0, 100.0, 0.5):
                time = round(self.disturbance(time_random(300)))
                distance = round(self.disturbance(self.linear_equation(i, time)), 2)
                self.data.add("%s,%s,%s\n" % (i, time, distance))

            for line in self.data:
                f.write(line)


def time_random(upper_bound):
    return random.uniform(0, upper_bound)
