import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly

from .linear_reg import LinearRegression


# plotly.offline.init_notebook_mode()

def multivariate_linear_reg(data_path, attr, num_iterations=1000, learning_rate=0.01):
    data = pd.read_csv(data_path)

    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name_1, input_param_name_2, output_param_name = attr

    x_train = train_data[[input_param_name_1, input_param_name_2]].values
    y_train = train_data[[output_param_name]].values

    x_test = test_data[[input_param_name_1, input_param_name_2]].values
    y_test = test_data[[output_param_name]].values

    plot_training_trace = go.Scatter3d(
        x=x_train[:, 0].flatten(),
        y=x_train[:, 1].flatten(),
        z=y_train.flatten(),
        name="Training Set",
        mode="markers",
        marker={
            "size": 10,
            "opacity": 1,
            "line": {
                "color": "rgb(255,255,255)",
                "width": 1
            },
        }
    )

    plot_test_trace = go.Scatter3d(
        x=x_test[:, 0].flatten(),
        y=x_test[:, 1].flatten(),
        z=y_test.flatten(),
        name="Test Set",
        mode="markers",
        marker={
            "size": 10,
            "opacity": 1,
            "line": {
                "color": "rgb(255,255,255)",
                "width": 1
            },
        }
    )

    plot_layout = go.Layout(
        title="Data Sets",
        scene={
            "xaxis": {"title": input_param_name_1},
            "yaxis": {"title": input_param_name_2},
            "zaxis": {"title": output_param_name},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0}
    )

    plot_data = [plot_training_trace, plot_test_trace]

    plot_figure = go.Figure(data=plot_data, layout=plot_layout)

    plotly.offline.plot(plot_figure)

    polynomial_degree = 0
    sinusoid_degree = 0

    linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
    theta, cost_history = linear_regression.train(learning_rate, num_iterations)

    print("初始损失：%s" % cost_history[0])
    print("最终损失：%s" % cost_history[-1])

    plt.plot(range(num_iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Progress")
    plt.show()

    predictions_num = 10

    x_min = x_train[:, 0].min()
    x_max = x_train[:, 0].max()

    y_min = x_train[:, 1].min()
    y_max = x_train[:, 1].max()

    x_axis = np.linspace(x_min, x_max, predictions_num)
    y_axis = np.linspace(y_min, y_max, predictions_num)

    x_predictions = np.zeros((predictions_num * predictions_num, 1))
    y_predictions = np.zeros((predictions_num * predictions_num, 1))

    x_y_index = 0

    for x_index, x_value in enumerate(x_axis):
        for y_index, y_value in enumerate(y_axis):
            x_predictions[x_y_index] = x_value
            y_predictions[x_y_index] = y_value
            x_y_index += 1

    z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

    plot_predictions_trace = go.Scatter3d(
        x=x_predictions.flatten(),
        y=y_predictions.flatten(),
        z=z_predictions.flatten(),
        name="Prediction Plane",
        mode="markers",
        marker={
            "size": 1
        },
        opacity=0.8,
        surfaceaxis=2
    )

    plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
    plot_figure = go.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(plot_figure)
