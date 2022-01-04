import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses, models

from Custom.WindowGenerator import WindowGenerator

tf.random.set_seed(42)
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

# # Window generator creation function


def create_window_generator(
    subject=None, input_width=20, shift=3, label_width=1, batch_size=64, features=["RMS"], add_knee=False, out_labels=["ankle moment"]
):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # #Get subject weight.
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    w = subject_details[f"S{subject}"]["weight"]
    # #Get trials directory
    trials = ["train_01", "train_02", "val", "test"]
    trials = list(
        map(lambda x: f"../Dataset/S{subject}/{x}_dataset.csv", trials))
    # #Load data
    train_01_df = pd.read_csv(trials[0], index_col="time")
    train_02_df = pd.read_csv(trials[1], index_col="time")
    val_df = pd.read_csv(trials[2], index_col="time")
    test_df = pd.read_csv(trials[3], index_col="time")
    # #Prepare scalers
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaler.fit(train_01_df.dropna().iloc[:100, :-8])
    angle_scaler = MinMaxScaler(feature_range=(0, 1))
    angle_scaler.fit(train_01_df.dropna().iloc[:100, -8:-4])
    # #Scale the dataset
    for data in [train_01_df, train_02_df, val_df, test_df]:
        data = scale_function(
            data, weight=w, features_scaler=features_scaler, angle_scaler=angle_scaler
        )

    train_01_df = choose_features(train_01_df, features=features)
    train_02_df = choose_features(train_02_df, features=features)
    val_df = choose_features(val_df, features=features)
    test_df = choose_features(test_df, features=features)
    # #Create Window object
    window_object = WindowGenerator(
        train_01_df,
        train_02_df,
        val_df,
        test_df,
        input_width=input_width,
        shift=shift,
        label_width=label_width,
        batch_size=batch_size,
        add_knee=add_knee,
        out_labels=out_labels,
    )
    return window_object


# #models
def create_lstm_model(window_object):
    custom_LSTM = partial(layers.LSTM, units=4, dropout=0.3)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            # custom_LSTM(4, return_sequences=True),
            custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    return lstm_model


def create_single_lstm_model(window_object):
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    custom_LSTM = partial(layers.LSTM, dropout=0.2)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            # custom_LSTM(16, return_sequences=True),
            # custom_LSTM(16, return_sequences=True),
            custom_LSTM(256, return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    return lstm_model


def create_conv_model(window_object):
    conv_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            # layers.BatchNormalization(),
            layers.Conv1D(filters=20, kernel_size=3,
                          strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
            layers.Conv1D(filters=30, kernel_size=3,
                          strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
            layers.Conv1D(filters=window_object.out_nums,
                          kernel_size=1, strides=1),
        ]
    )
    return conv_model


def create_nn_model(window_object):
    custom_nn = partial(layers.Dense, units=16, activation='relu')
    nn_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            layers.Flatten(),
            custom_nn(),
            custom_nn(),
            custom_nn(),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    return nn_model


def create_nn_gm_model(window_object):
    nn_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            layers.Flatten(),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
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
    rmse = partial(metrics.mean_squared_error, squared=False)
    error = []
    max_error = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        error.append(rmse(y_true_col, y_pred_col))
        max_error.append(metrics.max_error(y_true_col, y_pred_col))
    return np.around(error, 3), np.around(max_error, 3)


def normalized_rmse(y_true, y_pred):
    error = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        nmse = np.sum(np.square(y_true_col-y_pred_col))/np.sum(np.square(y_true_col))
        error.append(np.sqrt(nmse))
    return np.around(error, 3)


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
        squared_loss_for_positive = tf.math.scalar_mul(
            self.threshold, squared_loss)
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


def plot_results(y_true, y_pred, out_labels, R2_score, rmse_result, max_error, folder):
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure("Prediction", figsize=(11, 9))
    time = [i / 20 for i in range(len(y_true))]
    for i, col in enumerate(list(out_labels)):
        plt.subplot(len(out_labels), 1, i + 1)
        print(f"{col} R2 score: {R2_score[i]}")
        print(f"{col} RMSE result: {rmse_result[i]}")
        print(f"{col} max error is {max_error}Nm/Kg")
        plt.plot(time, -y_true[:, i], linewidth=2.5)
        plt.plot(time, -y_pred[:, i], "r--", linewidth=2,)
        plt.title(col, fontsize=title_size)
        if i == 0:
            plt.legend(["measured moment", "prediction"], fontsize=label_size)
        plt.xlim((0, 9))
        plt.xlabel("Time [s]", fontsize=label_size)
        if "moment" in col:
            plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
        else:
            plt.ylabel("Angle [Degree]", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}{out_labels[i]}.svg")
    plt.savefig(f"{folder}{out_labels[i]}.pdf")
    plt.draw()


def plot_models(predictions: dict, y_true, path: str, subject=None):
    time = [i / 20 for i in range(len(y_true))]
    # fig, ax = plt.subplots(nrows=len(predictions.keys()))
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure(f"S{subject} GM", figsize=(11, 9))
    for i, model_name in enumerate(predictions.keys()):
        plt.subplot(len(predictions.keys()), 1, i+1)
        plt.plot(time, -y_true, linewidth=2.5, label="measured moment")
        plt.plot(time, -predictions[model_name],
                 "r--", linewidth=2, label="prediction")
        plt.title(model_name, fontsize=title_size)
        plt.xlim((0, 9))
        plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
        plt.yticks([0, 0.5, 1, 1.5], fontsize=tick_size)
        plt.ylim([-0.25, 1.52])
        plt.grid(True)
    plt.xticks(fontsize=tick_size)
    plt.xlabel("Time [s]", fontsize=label_size)
    plt.legend(bbox_to_anchor=(1, -0.5), loc="lower right",
               borderaxespad=0, ncol=2, fontsize=label_size)
    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{path}S{subject}_models_results.pdf")
