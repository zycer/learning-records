from sklearn import preprocessing as sp
from sklearn import pipeline as pl
from sklearn import linear_model as lm
from sklearn import metrics as sm
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly


def multivariate_mul_regression(data_path, attr):
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

