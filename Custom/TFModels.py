from functools import partial
from tensorflow.keras import layers, models


def create_lstm_model(window_object, stacked=3):
    custom_LSTM = partial(layers.LSTM, units=8, dropout=0.1)
    
    # x_input = layers.Input(shape=(window_object.input_width,
    #                         window_object.features_num))
    # for i in range(stacked-1):
    #     if i == 0:
    #         x = custom_LSTM(return_sequences=True)(x_input)
    #     else:
    #         x = custom_LSTM(return_sequences=True)(x)
    # x = custom_LSTM(return_sequences=False)(x)
        
    # x = layers.Dense(window_object.out_nums * window_object.label_width)(x)
    # x = layers.Reshape([window_object.label_width,
    #                     window_object.out_nums])(x)
    # lstm_model = models.Model(inputs=x_input, outputs=x, name="lstm_model")
    # lstm_model.summary()
    # return lstm_model

    lstm_model = models.Sequential(
        [
            layers.InputLayer((window_object.input_width,
                              window_object.features_num)),
            # custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=True),
            custom_LSTM(return_sequences=False),
            layers.Dense(window_object.out_nums * window_object.label_width),
            layers.Reshape(
                [window_object.label_width, window_object.out_nums]),
        ]
    )
    lstm_model.summary()
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
    custom_nn = partial(layers.Dense, units=8, activation='selu')
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
