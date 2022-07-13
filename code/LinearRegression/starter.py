import random

from random_data2 import EV
from MultivariateLinearReg import multivariate_linear_reg
from multinomial_regression import multinomial_reg
from univariate_linear_reg import linear_reg
from multivariate_multinomial_regression import multivariate_mul_regression

if __name__ == "__main__":
    # 线性
    # ev_1 = EV(name="ev_1", k=5, b=5.2, t_discount=0.46)
    # ev_1.generate_ev_data(reg_type="linear")
    # multivariate_linear_reg(ev_1.file_path, ev_1.attribute, 1000)

    # 非线性
    # ev_2 = EV(name="ev_2", k=4, b=3.2, t_discount=0.46)
    # ev_2.generate_ev_data(reg_type="non_linear")
    # multinomial_reg(ev_2.file_path, ev_2.attribute)

    ev_3 = EV(name="ev_3", base_number=1.024, battery_health=100,
              temperature=20, ev_weight=3000, load=0, standard_distance_index=60)
    ev_3.generate_ev_data(200)
    linear_reg(ev_3.data_path, ("weight", "distance"))
    multinomial_reg(ev_3.data_path, ("soc", "battery_health", "distance"))
    multivariate_mul_regression(ev_3.data_path, ev_3.effect_data_path, ev_3.attribute)
