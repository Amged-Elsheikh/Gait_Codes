import tensorflow as tf
from sklearn import metrics
from functools import partial
from tensorflow.keras import losses
import numpy as np


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
    nrmse = []
    max_error = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        error.append(rmse(y_true_col, y_pred_col))
        max_error.append(np.max(np.abs(y_true_col - y_pred_col)))
        delta = max(y_true_col) - min(y_true_col)
        nrmse.append(error[i]/delta)
    return np.around(error, 3), np.around(nrmse,3), np.around(max_error, 3)



def normalized_rmse(y_true, y_pred):
    error = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        nmse = np.sum(np.square(y_true_col-y_pred_col)) / (np.max(y_true_col)-np.min(y_true_col))
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
        if self.threshold == 0:
            self.threshold=1

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