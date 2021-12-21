import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from functools import partial
import tensorflow as tf
from tensorflow.keras import layers, models, losses

"""
This python file will contains custm function for preparing models, tf models, custom losses and custom evaluation functions.
"""

# # Data preparation
def scale_function(data, weight, features_scaler, angle_scaler):
    """This function will scale the dataset, if you do not want to scale something, set it's value to None.

    Args:
        data ([pandas dataframe]): [The data columns are arranged as fellow: EMG features, angles and moments]
        weight ([float or None]): [subject weight. If None, no scale will be performed.]
        features_scaler ([sci-kit learn scaler]): [Scale the features. Use MinMax scaler. If None, no scale will be performed.]
        angle_scaler ([sci-kit learn scaler]): [Scale the angles. Use MinMax scaler. If None, no scale will be performed.]
    """
    # Scale moments
    if weight:
        data.iloc[:, -4:] = data.iloc[:, -4:] / weight
    # Scale features
    if features_scaler:
        data.iloc[:, :-8] = features_scaler.transform(data.iloc[:, :-8])
    # Scale angles
    if angle_scaler:
        columns = data.columns[-8:-4]
        data[columns] = angle_scaler.transform(data[columns])
    return


def choose_features(data, features=["RMS"]):
    new_columns = []
    for col in data.columns:
        if "DEMG" in col:  # Confirm the column is a feature column
            for feature in features:
                if feature.lower() in col.lower():
                    new_columns.append(col)
                    continue
        else:  # append all labels
            new_columns.append(col)
    return data[new_columns]


# #models
def create_lstm_model(window_object):
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    custom_LSTM = partial(layers.LSTM, dropout=0.3,)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width, window_object.features_num)),
            # layers.BatchNormalization(),
            # custom_LSTM(4, return_sequences=True),
            custom_LSTM(4, return_sequences=True),
            custom_LSTM(4, return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape([window_object.label_width, window_object.out_nums]),
        ]
    )
    return lstm_model


def create_conv_model(window_object):
    conv_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width, window_object.features_num)),
            # layers.BatchNormalization(),
            layers.Conv1D(filters=20, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
            layers.Conv1D(filters=30, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
            layers.Conv1D(filters=window_object.out_nums, kernel_size=1, strides=1),
        ]
    )
    return conv_model


def create_nn_model(window_object):
    nn_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width, window_object.features_num)),
            layers.Flatten(),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape([window_object.label_width, window_object.out_nums]),
        ]
    )
    return nn_model


# # Evaluation functions
def nan_R2(y_true, y_pred):
    R2 = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        R2.append(metrics.r2_score(y_true_col, y_pred_col))
    return np.around(R2, 4)


def nan_rmse(y_true, y_pred):
    mse = partial(metrics.mean_squared_error, squared=False)
    rmse = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        rmse.append(mse(y_true_col, y_pred_col))
    return np.around(rmse, 3)


def custom_loss(y_true, y_pred):
    error = tf.reshape(y_true - y_pred, (-1, 1))
    error = error[~tf.math.is_nan(error)]
    return tf.reduce_mean(tf.square(error), axis=0)


class SPLoss(losses.Loss):  # Slip Prevention Loss
    def __init__(self, threshold=3.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_positive = y_true > 0
        squared_loss = tf.square(error) / 2
        squared_loss_for_positive = tf.math.scalar_mul(self.threshold, squared_loss)
        return tf.where(is_positive, squared_loss_for_positive, squared_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# #Plotting functions
def plot_learning_curve(history, folder):
    if history == None:
        print("No train history was found")
    else:
        plt.figure("Learning curve")
        plt.plot(
            history.epoch,
            history.history["loss"],
            history.epoch,
            history.history["val_loss"],
        )
        plt.legend(["train loss", "val loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.draw()
        plt.savefig(f"{folder}learning_curve.pdf")


def plot_results(y_true, y_pred, out_labels, R2_score, rmse_result, folder):
    time = [i / 20 for i in range(len(y_true))]
    plt.figure("Prediction")
    for i, col in enumerate(list(out_labels)):
        plt.subplot(len(out_labels), 1, i + 1)
        print(f"{col} R2 score: {R2_score[i]}")
        print(f"{col} RMSE result: {rmse_result[i]}")
        plt.plot(time, y_true[:, i], linewidth=2.5)
        plt.plot(
            time, y_pred[:, i], "r--", linewidth=2.5,
        )
        plt.title(col)
        if i == 0:
            plt.legend(["y_true", "y_pred"], loc="lower left")
        plt.xlim((time[-600], time[-100]))
        plt.xlabel("Time [s]")
        if "moment" in col:
            plt.ylabel("Moment [Nm]")
        else:
            plt.ylabel("Angle [Degree]")
    plt.tight_layout()
    plt.savefig(f"{folder}{out_labels[i]}.svg")
    plt.savefig(f"{folder}{out_labels[i]}.pdf")
    plt.draw()
