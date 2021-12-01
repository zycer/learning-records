import random

from random_data2 import EV
from LinearRegression.MultivariateLinearReg import multivariate_linear_reg
from LinearRegression.multinomial_regression import multinomial_reg
from LinearRegression.univariate_linear_reg import linear_reg

if __name__ == "__main__":
    # 线性
    # ev_1 = EV(name="ev_1", k=5, b=5.2, t_discount=0.46)
    # ev_1.generate_ev_data(reg_type="linear")
    # multivariate_linear_reg(ev_1.file_path, ev_1.attribute, 1000)

    # 非线性
    # ev_2 = EV(name="ev_2", k=4, b=3.2, t_discount=0.46)
    # ev_2.generate_ev_data(reg_type="non_linear")
    # multinomial_reg(ev_2.file_path, ev_2.attribute)

    ev_3 = EV("ev_3", 1.024, 100, 20, 3000, 0, 60)
    ev_3.generate_ev_data(100)
    # linear_reg(ev_3.file_path, ("weight", "distance"))

    multinomial_reg(ev_3.file_path, ("soc", "battery_health", "distance"))
