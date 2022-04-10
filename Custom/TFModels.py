from functools import partial
from tensorflow.keras import layers, models


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


def create_ff_model(window_object):
    custom_nn = partial(layers.Dense, units=4, activation='relu')
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
