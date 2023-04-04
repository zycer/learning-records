import json

from sklearn import preprocessing as sp
from sklearn import pipeline as pl
from sklearn import linear_model as lm
from sklearn import metrics as sm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def multivariate_mul_regression(data_path, effect_data_path, attr):
    data = pd.read_csv(data_path)
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name_1, input_param_name_2, input_param_name_3, input_param_name_4, output_param_name = attr

    x_train = train_data[[input_param_name_1, input_param_name_2, input_param_name_3, input_param_name_4]].values
    y_train = train_data[[output_param_name]].values

    x_test = test_data[[input_param_name_1, input_param_name_2, input_param_name_3, input_param_name_4]].values
    y_test = test_data[[output_param_name]].values

    model = pl.make_pipeline(
        sp.PolynomialFeatures(degree=3, include_bias=False),
        lm.LinearRegression()
    )

    param_grid = {"polynomialfeatures__degree": np.arange(8),
                  "linearregression__fit_intercept": [True, False],
                  "linearregression__normalize": [True, False]}

    grid = GridSearchCV(model, param_grid, cv=7)

    grid.fit(x_train, y_train)
    print("最佳模型系数: ")
    print(grid.best_params_)
    print()

    best_model = grid.best_estimator_

    best_model.fit(x_train, y_train)
    z_predictions = best_model.predict(x_test)

    print("特征系数矩阵：")
    print(best_model.steps[1][1].coef_)
    print()
    print(f"回归方程截距：{best_model.steps[1][1].intercept_}\n")
    print("特征对应关系：")
    for idx, att in enumerate(attr[:-1]):
        print(f"{att}: x{idx + 1}")
    print("回归方程：")
    print("F(x)="
          f"{best_model.steps[1][1].intercept_}+{best_model.steps[1][1].coef_[0][0]}x1+"
          f"{best_model.steps[1][1].coef_[0][1]}x2+{best_model.steps[1][1].coef_[0][2]}x3+"
          f"{best_model.steps[1][1].coef_[0][3]}x4+{best_model.steps[1][1].coef_[0][4]}x1^2+"
          f"{best_model.steps[1][1].coef_[0][5]}x1x2+{best_model.steps[1][1].coef_[0][6]}x1x3+"
          f"{best_model.steps[1][1].coef_[0][7]}x1x4+{best_model.steps[1][1].coef_[0][8]}x2^2+"
          f"{best_model.steps[1][1].coef_[0][9]}x2x3+{best_model.steps[1][1].coef_[0][10]}x2x4+"
          f"{best_model.steps[1][1].coef_[0][11]}x3^2+{best_model.steps[1][1].coef_[0][12]}x3x4+"
          f"{best_model.steps[1][1].coef_[0][13]}x4^2")

    print()
    print("平均绝对误差：", sm.mean_absolute_error(y_test, z_predictions))
    print("均方误差：", sm.mean_squared_error(y_test, z_predictions))
    print("绝对中位差：", sm.median_absolute_error(y_test, z_predictions))
    print("R2_得分：", sm.r2_score(y_test, z_predictions))
    print()

    # error_map = zip(z_predictions, y_test)
    # for i in error_map:
    #     print(i)

    with open(effect_data_path, "r") as f:
        effect_data: dict = json.load(f)

    for key, value in effect_data.items():
        x_data, y_data = [], []
        for item_key, item_value in value.items():
            if item_key == "key":
                input_param_name, output_param_name = item_value
                plt.xlabel(input_param_name)
                plt.ylabel(output_param_name)
            else:
                x_data = [i[0] for i in item_value]
                y_data = [i[1] for i in item_value]

        plt.scatter(x_data, y_data, label="Train data")

        plt.title("Relationship between SOC and distance")
        plt.legend()
        plt.show()
