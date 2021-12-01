import random
import numpy as np


class EV:
    attribute = [
        "soc",
        "weight",
        "temperature",
        "battery_health",
        "distance"
    ]

    def __init__(self, name, base_number, battery_health, temperature, ev_weight, load, standard_distance_index):
        """
        初始化
        :param name: EV编号
        :param a:
        :param battery_health: 电池健康度
        :param temperature: 温度
        :param ev_weight: EV空载重量（kg）
        :param load: EV负载重量（kg）
        :param standard_distance_index: EV标准续航系数
        """
        self.base_number = base_number
        self.soc = 0
        self.name = name
        self.load = load
        self.ev_weight = ev_weight
        self.temperature = temperature
        self.battery_health = battery_health
        self.standard_distance_index = standard_distance_index
        self.file_path = f"./data/{self.name}.csv"
        self.data = set()

        self.standard_distance = self.ideal_performance

    @property
    def effect_battery_health_soc(self):
        return self.soc * (1 - self.battery_health / 100)

    @property
    def effect_temperature_soc(self):
        """
        温度对电池SOC的影响
        :return: 影响值（百分比）
        """
        return 1.08 ** -self.temperature - 1.5

    @property
    def effect_weight_distance(self):
        """
        EV负载重量对续航里程的影响
        :return: 影响值(百分比)
        """
        a = 0.02
        b = 1 - (a * self.ev_weight + 50 * a)
        return a * (self.ev_weight + self.load) + b

    @property
    def ideal_performance(self):
        """
        理想续航里程
        :return:里程数（km）
        """
        return (self.base_number ** self.soc - 1) * self.standard_distance_index

    @property
    def comprehensive_performance(self):
        """
        综合里程数
        :return: 受到温度/载重影响后的续航里程数
        """
        soc = self.soc - self.effect_temperature_soc - self.effect_battery_health_soc
        self.soc = round(soc, 2) if soc > 0 else 0
        self.soc = self.soc if self.soc <= 100 else 100
        performance = self.ideal_performance
        performance -= performance * self.effect_weight_distance / 100
        return round(performance, 2)

    @classmethod
    def __disturbance(cls, value, coefficient, rate, noise=0.1, positive=True):
        """
        扰动函数
        :param value: 被扰动值
        :param coefficient: 扰动系数
        :param rate: 扰动幅度
        :param noise: 噪声比
        :param positive: 是否为正
        :return: 扰动后的值
        """
        ran = random.uniform(0, 1)
        if ran < noise:
            figure = 2
            value += random.uniform(-figure * coefficient * rate, figure * coefficient * rate)
        else:
            value += random.uniform(-coefficient * rate, coefficient * rate)

        if positive:
            return value if value > 0 else 0
        return value

    @classmethod
    def random_value(cls, upper_limit, lower_limit):
        return round(random.uniform(upper_limit, lower_limit), 2)

    def generate_ev_data(self, data_count):
        with open(self.file_path, "w") as f:
            f.write(','.join(self.attribute) + "\n")
            for i in range(data_count):
                self.soc = self.random_value(0, 100)
                self.battery_health = self.random_value(80, 100)
                self.load = self.random_value(0, 1000)
                self.temperature = self.random_value(-40, 60)
                performance = self.comprehensive_performance
                self.data.add("%s,%s,%s,%s,%s\n" % (
                    self.soc, round(self.ev_weight + self.load, 2), self.temperature, self.battery_health, performance))

            for line in self.data:
                f.write(line)



