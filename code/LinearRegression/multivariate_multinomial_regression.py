import json

from sklearn import preprocessing as sp
from sklearn import pipeline as pl
from sklearn import linear_model as lm
from sklearn import metrics as sm
import pandas as pd
import matplotlib.pyplot as plt


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
        sp.PolynomialFeatures(8),
        lm.LinearRegression()
        # lm.Ridge(10, fit_intercept=True, max_iter=1000)
    )

    model.fit(x_train, y_train)
    z_predictions = model.predict(x_test)

    print(sm.mean_absolute_error(y_test, z_predictions))
    print(sm.mean_squared_error(y_test, z_predictions))
    print(sm.median_absolute_error(y_test, z_predictions))
    print(sm.r2_score(y_test, z_predictions))

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

