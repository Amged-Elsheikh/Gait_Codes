from functools import partial

import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)


def select_GPU(gpu_index=0):
    """
    This function will  be used to select which GPU to be used.
    """
    # Insure CUDA is available
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    print('Have cuda')
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    print(gpus)
    # Select the GPU
    tf.config.experimental.set_visible_devices(gpus[gpu_index], "GPU")
    # Prevent script from booking all available resources. If you removed
    # the followng line, GPU will run on script at the time.
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    return


def create_lstm_model(window):
    custom_LSTM = partial(layers.LSTM, units=8, dropout=0.1)
    lstm_model = models.Sequential(
        [
            layers.InputLayer((window.input_width, window.features_num)),
            custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=False),
            layers.Dense(window.out_nums * window.label_width),
            layers.Reshape([window.label_width, window.out_nums]),
        ]
    )
    lstm_model.summary()
    return lstm_model


def create_conv_model(window):
    conv_model = models.Sequential(
        [
            layers.InputLayer((window.input_width, window.features_num)),
            layers.Conv1D(filters=16, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=3, strides=1, padding="valid"),
            layers.LSTM(4, return_sequences=True),
            layers.LSTM(4, return_sequences=False),
            layers.Dense(window.out_nums * window.label_width),
            layers.Reshape([window.label_width, window.out_nums]),
        ]
    )
    conv_model.summary()
    return conv_model


def create_ff_model(window):
    custom_nn = partial(layers.Dense, units=4, activation="selu")
    nn_model = models.Sequential(
        [
            layers.InputLayer((window.input_width, window.features_num)),
            layers.Flatten(),
            custom_nn(),
            custom_nn(),
            custom_nn(),
            layers.Dense(window.out_nums * window.label_width),
            layers.Reshape([window.label_width, window.out_nums]),
        ]
    )
    nn_model.summary()
    return nn_model


def model_callbacks(model_file: str):
    checkpoint_callback = ModelCheckpoint(
        filepath=model_file,
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", min_delta=1e-3, factor=0.7, patience=20
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
    return [checkpoint_callback, reduce_lr, early_stop]


def evaluate(y_true: np.array, y_pred: np.array):
    """
    Get R2 score. This function is customed to handle missing values
    """
    R2 = []
    RMSE = partial(metrics.mean_squared_error, squared=False)
    rmse_error = []
    nrmse = []
    joints_num = np.shape(y_true)[1]
    for joint in range(joints_num):
        y_true_col = y_true[:, joint]
        y_pred_col = y_pred[:, joint]
        # Create a logic to remove Nan
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        # Calculate and append the R2
        delta = max(y_true_col) - min(y_true_col)
        R2.append(metrics.r2_score(y_true_col, y_pred_col))
        rmse_error.append(RMSE(y_true_col, y_pred_col))
        nrmse.append(rmse_error[joint] / delta)
    return np.around(R2, 4), np.around(rmse_error, 3), np.around(nrmse, 3)