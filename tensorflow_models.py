import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from functools import partial
import tensorflow as tf
from tensorflow.keras import layers, models
"""
This python file will contains tf models, custom losses and custom evaluation functions.
"""
# #models


def create_lstm_model(window_object):
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    custom_LSTM = partial(layers.LSTM, dropout=0.3,)
    lstm_model = models.Sequential([
        layers.InputLayer((window_object.input_width,
                          window_object.features_num)),
        # layers.BatchNormalization(),
        # custom_LSTM(4, return_sequences=True),
        custom_LSTM(4, return_sequences=True),
        custom_LSTM(4, return_sequences=False),
        layers.Dense(window_object.out_nums * window_object.label_width),
        layers.Reshape([window_object.label_width, window_object.out_nums])
    ])
    return lstm_model


def create_conv_model(window_object):
    conv_model = models.Sequential([
        layers.InputLayer((window_object.input_width,
                          window_object.features_num)),
        # layers.BatchNormalization(),
        layers.Conv1D(filters=20, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
        layers.Conv1D(filters=30, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=2, strides=1, padding="valid"),
        layers.Conv1D(filters=window_object.out_nums, kernel_size=1, strides=1)
    ])
    return conv_model


def create_nn_model(window_object):
    nn_model = models.Sequential([
        layers.InputLayer((window_object.input_width,
                          window_object.features_num)),
        layers.Flatten(),
        layers.Dense(32),
        layers.Dense(32),
        layers.Dense(32),
        layers.Dense(window_object.out_nums * window_object.label_width),
        layers.Reshape([window_object.label_width, window_object.out_nums])
    ])
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
    return np.around(rmse, 2)


def custom_loss(y_true, y_pred):
    error = tf.reshape(y_true - y_pred, (-1, 1))
    error = error[~tf.math.is_nan(error)]
    return tf.reduce_mean(tf.square(error), axis=0)


def plot_learning_curve(history, folder):
    if history == None:
        print("No train history was found")
        return None
    else:
        plt.figure("Learning curve")
        plt.plot(history.epoch, history.history['loss'],
                 history.epoch, history.history['val_loss'])
        plt.legend(["train loss", "val loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.draw()
        plt.savefig(f"{folder}learning_curve.pdf")
        plt.close()
        return None
