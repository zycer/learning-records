import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_reg import LinearRegression


def linear_reg(data_path, attr):
    data = pd.read_csv(data_path)
    # 得到训练和测试数据
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name, output_param_name = attr

    x_train = train_data[[input_param_name]].values
    y_train = train_data[[output_param_name]].values

    x_test = test_data[input_param_name].values
    y_test = test_data[output_param_name].values

    plt.scatter(x_train, y_train, label="Train data")
    plt.scatter(x_test, y_test, label="Test data")
    plt.xlabel(input_param_name)
    plt.xlabel(output_param_name)
    plt.title("Relationship between SOC and distance")
    plt.legend()
    plt.show()

    num_iteration = 500
    learn_rate = 0.01

    linear_regression = LinearRegression(x_train, y_train)
    theta, cost_history = linear_regression.train(learn_rate, num_iteration)

    print("开始时的损失：%s" % cost_history[0])
    print("训练后的损失：%s" % cost_history[-1])

    plt.plot(range(num_iteration), cost_history)
    plt.xlabel("Iter")
    plt.ylabel("cost")
    plt.title("GD")
    plt.show()

    predictions_num = 100
    x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num,  1)
    y_predictions = linear_regression.predict(x_predictions)

    plt.scatter(x_train, y_train, label="Train data")
    plt.scatter(x_test, y_test, label="Test data")
    plt.plot(x_predictions, y_predictions, 'r', label="Prediction")
    plt.xlabel(output_param_name)
    plt.xlabel(input_param_name)
    plt.title("Relationship between SOC and distance")
    plt.legend()
    plt.show()