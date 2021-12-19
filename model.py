# import matplotlib as mpl
from Custom.WindowGenerator import WindowGenerator
from functools import partial
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow_models import *
K = keras.backend
# import os
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# mpl.rcParams['figure.dpi'] = 110


# Scaling function

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
        data.iloc[:, -4:] = data.iloc[:, -4:]/weight
    # Scale features
    if features_scaler:
        data.iloc[:, :-8] = features_scaler.transform(data.iloc[:, :-8])
    # Scale angles
    if angle_scaler:
        columns = data.columns[-8:-4]
        data[columns] = angle_scaler.transform(data[columns])
    return


def plot_results(window_object, y_true, y_pred, R2_score, rmse_result, folder):
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
        if i == 0:
            plt.legend(["y_true", "y_pred"], loc='lower left')
        plt.xlim((time[-600], time[-100]))
        plt.xlabel("Time [s]")
        if "moment" in col:
            plt.ylabel("Moment [Nm]")
        else:
            plt.ylabel("Angle [Degree]")
    plt.tight_layout()
    plt.savefig(f"{folder}{labels[i]}.svg")
    plt.savefig(f"{folder}{labels[i]}.pdf")
    plt.draw()
    plt.close()


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


def create_window_generator(subject=None):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # #Get subject weight.
    w = subject_details[f'S{subject}']['weight']
    # #Get trials directory
    trials = ["train_01", "train_02", "val", "test"]
    trials = list(
        map(lambda x: f"../Dataset/S{subject}/{x}_dataset.csv", trials))
    # #Load data
    train_01_df = pd.read_csv(trials[0], index_col='time'),
    train_02_df = pd.read_csv(trials[1], index_col='time')
    val_df = pd.read_csv(trials[2], index_col='time')
    test_df = pd.read_csv(trials[3], index_col='time')
    # #Prepare scalers
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaler.fit(train_01_df.iloc[:400, :-8])
    angle_scaler = MinMaxScaler(feature_range=(0, 1))
    angle_scaler.fit(train_01_df.iloc[:400, -8:-4])
    # #Scale the dataset
    for data in [train_01_df, train_02_df, val_df, test_df]:
        data = scale_function(
            data, weights=w, features_scaler=features_scaler, angle_scaler=angle_scaler)

    train_01_df = choose_features(train_01_df, features=features)
    train_02_df = choose_features(train_02_df, features=features)
    val_df = choose_features(val_df, features=features)
    test_df = choose_features(test_df, features=features)
    # #Create Window object
    window_object = WindowGenerator(train_01_df, train_02_df, val_df, test_df,
                                    input_width=15, shift=3, label_width=1,
                                    batch_size=64, add_knee=add_knee, out_labels=out_labels)
    return window_object


def train_fit(subject, tested_on, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False):

    window_object = create_window_generator(subject)
    if tested_on == None:
        tested_on = subject

    # setup model folder
    global folder
    folder = f"../Results/indiviuals/{model_name}/S{subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    w = subject_details[f'S{subject}']['weight']
    # Get all dataset
    train_set, val_set, _ = window_object.make_dataset()
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
            plot_learning_curve(history, folder)
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
    window_object = create_window_generator(tested_on)
    _, _, test_set = window_object.make_dataset()
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]

    ################ Evaluation and plot ################
    r2_score = nan_R2(y_true, y_pred)
    rmse_result = nan_rmse(y_true, y_pred)
    folder = f"../Results/indiviuals/{model_name}/S{tested_on}/"
    plot_results(window_object, y_true, y_pred, r2_score, rmse_result, folder)
    plt.draw()
    return history, y_true, y_pred, r2_score, rmse_result


if __name__ == '__main__':
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    features = ["RMS", "ZC"]
    add_knee = False
    out_labels = ["ankle moment"]
    model_name = "nn_model"

    model_dic = {}

    model_dic["lstm_model"] = create_lstm_model
    model_dic["conv_model"] = create_conv_model
    model_dic["nn_model"] = create_nn_model

    # w1 = create_window_generator(subject="1")
    # w2 = create_window_generator(subject="2")
    # w4 = create_window_generator(subject="4")

    # Train and test new/existing models
    for model_name in model_dic.keys():
        history, y_true, y_pred, r2, rmse = train_fit(subject="02", tested_on="04",
                                                      model_name=model_name, epochs=1000,
                                                      eval_only=True, load_best=True)
        print(model_name)
        # plt.show()
