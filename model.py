from numpy.lib.shape_base import column_stack
from Custom.WindowGenerator import WindowGenerator
# import matplotlib as mpl
from functools import partial
from sklearn import metrics
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
K = keras.backend

# mpl.rcParams['figure.dpi'] = 110

model_dic = {}

if not tf.test.is_built_with_cuda():
    raise print("No GPU found")

with open("subject_details.json", "r") as f:
    subject_details = json.load(f)


# Scaling functions
def scale_moment(data, weight, scale=True):
    if scale:
        data.iloc[:, -4:] = data.iloc[:, -4:]/weight
    return data


def scale_angle(data, angle_scaler, scale=True):
    if scale:
        columns = data.columns[-8:-4]
        data[columns] = angle_scaler.transform(data[columns])
    return data


def scale_features(data, scaler):
    data.iloc[:, :-8] = scaler.transform(data.iloc[:, :-8])
    return data

# window-object is a custom object


def make_dataset(window_object):
    train_set = window_object.get_train_dataset()
    val_set = window_object.get_val_dataset()
    test_set = window_object.get_evaluation_set()
    return train_set, val_set, test_set


def custom_loss(y_true, y_pred):
    error = tf.reshape(y_true - y_pred, (-1, 1))
    error = error[~tf.math.is_nan(error)]
    return tf.reduce_mean(tf.square(error), axis=0)

# Custom evaluation functions


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
    MSE = partial(metrics.mean_squared_error, squared=False)
    rmse = []
    _, l = np.shape(y_true)
    for i in range(l):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]
        logic = np.isfinite(y_true_col)
        y_true_col = y_true_col[logic]
        y_pred_col = y_pred_col[logic]
        rmse.append(MSE(y_true_col, y_pred_col))
    return np.around(rmse, 2)

# Plotting function


def plot_learning_curve(history):
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
        # plt.close()
        return None


def plot_results(window_object, y_true, y_pred, R2_score, rmse_result):
    global time
    time = [i/20 for i in range(len(y_true))]
    plt.figure("Prediction")
    labels = window_object.out_labels
    for i, col in enumerate(list(labels)):
        plt.subplot(len(labels), 1, i+1)
        print(f"{col} R2 score: {R2_score[i]}")
        print(f"{col} RMSE result: {rmse_result[i]}")
        plt.plot(time, y_true[:, i], linewidth=2.5)
        plt.plot(time, y_pred[:, i], "r--", linewidth=2.5,)
        plt.title(col)
        plt.legend(["y_true", "y_pred"])
        plt.xlim((time[-600], time[-100]))
        plt.xlabel("Time [s]")
        if "moment" in col:
            plt.ylabel("Moment [Nm]")
        else:
            plt.ylabel("Angle [Degree]")
    plt.savefig(f"{folder}{labels[i]}.pdf")
    plt.draw()
    # plt.close()

# Models
def create_lstm_model(window_object):
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    custom_LSTM = partial(layers.LSTM, dropout=0.3,)
    # kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2')
    lstm_model = keras.models.Sequential([
        layers.InputLayer((window_object.input_width,
                          window_object.features_num)),
        # layers.BatchNormalization(),
        # custom_LSTM(128, return_sequences=True),
        custom_LSTM(64, return_sequences=True),
        custom_LSTM(64, return_sequences=False),
        layers.Dense(window_object.out_nums * window_object.label_width),
        # window_object.out_nums
        layers.Reshape([window_object.label_width, window_object.out_nums])
    ])
    return lstm_model


def create_conv_model(window_object):
    conv_model = keras.models.Sequential([
        layers.InputLayer((window_object.input_width,
                          window_object.features_num)),
        layers.BatchNormalization(),
        layers.Conv1D(filters=10, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=20, kernel_size=3, strides=1, padding='same'),
        layers.Conv1D(filters=window_object.out_nums, kernel_size=1, strides=1)
    ])
    return conv_model


model_dic["lstm_model"] = create_lstm_model
model_dic["conv_model"] = create_conv_model


def train_fit(window_object, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False):
    # setup model folder
    global folder
    folder = f"../Results/indiviuals/{model_name}/S{subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get all dataset
    train_set, val_set, test_set = make_dataset(window_object)
    ##############################################################################################################
    # Load and compile new model
    K.clear_session()
    model = model_dic[model_name](window_object)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr),
                  loss="mean_squared_error")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{folder}S{subject}_{model_name}.hdf5",
        save_weights_only=True, monitor='val_loss',
        save_best_only=True)

    if load_best:
        try:
            model.load_weights(f"{folder}/S{subject}_{model_name}.hdf5")
        except:
            print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Train Model
    try:  # Train or load the best model the model
        if not eval_only:
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=[model_checkpoint_callback])
            plot_learning_curve(history)
            # Load the best model
            model.load_weights(f"{folder}/S{subject}_{model_name}.hdf5")
        else:
            history = "No training was conducted"
            model.load_weights(f"{folder}/S{subject}_{model_name}.hdf5")
    except KeyboardInterrupt:
        history = "\n\nTrains stopped manually"
        print(history)
    except OSError:  # If no saved model to be evaluated exist
        print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Get predictions and real values
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]
    ##############################################################################################################
    ## Evaluation and plot
    r2_score = nan_R2(w*y_true, w*y_pred)
    rmse_result = nan_rmse(w*y_true, w*y_pred)
    plot_results(window_object, y_true, y_pred, r2_score, rmse_result)
    plt.show()
    return history, y_true, y_pred, r2_score, rmse_result

# Import and scale the data


subject = "02"
w = subject_details[f'S{subject}']['weight']
dataset_folder = f"../Dataset/S{subject}/"
trials = ["train_01", "train_02", "val", "test"]
trials = list(map(lambda x: f"{dataset_folder}{x}_dataset.csv", trials))
train_01_df = scale_moment(pd.read_csv(trials[0], index_col='time'), weight=w)
train_02_df = scale_moment(pd.read_csv(trials[1], index_col='time'), weight=w)
val_df = scale_moment(pd.read_csv(trials[2], index_col='time'), weight=w)
test_df = scale_moment(pd.read_csv(trials[3], index_col='time'), weight=w)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_01_df.iloc[:400, :-8])
angle_scaler = MinMaxScaler(feature_range=(0, 1))
angle_scaler.fit(train_01_df.iloc[:400, -8:-4])
for data in [train_01_df, train_02_df, val_df, test_df]:
    data = scale_features(data, scaler)
    data = scale_angle(data, angle_scaler)

model_name = "lstm_model"
# Create Window object
w1 = WindowGenerator(train_01_df=train_01_df, train_02_df=train_02_df,
                     val_df=val_df, test_df=test_df, batch_size=64,
                     input_width=5, shift=1, label_width=1)
# Train and test new/existing models
history, y_true, y_pred, r2, rmse = train_fit(
    w1, model_name, epochs=3000, eval_only=True, load_best=False)
