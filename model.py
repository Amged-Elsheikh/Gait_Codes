import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tensorflow import keras

from Custom.DataHandler import DataHandler
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def add_mean_std(df):
    mean = df.mean()
    std = df.std()
    df.loc['mean',:] = mean
    df.loc['std',:] = std


# # Window generator creation function
def create_window_generator(
    subject=None, input_width=20, shift=3, label_width=1, batch_size=64, features=["RMS"], sensors=['sensor 1'], add_knee=False, out_labels=["ankle moment"]
):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # Get subject weight.
    dataHandler = DataHandler(subject, features, sensors, add_knee, out_labels)
    # #Create Window object
    window_object = WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size)
    return window_object

def model_callbacks(model_file):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file, save_weights_only=True,
        monitor="val_loss", save_best_only=True,)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                     min_delta=1e-3,
                                                     factor=0.8,  patience=20)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                  patience=50, restore_best_weights=True,)
    callbacks = [checkpoint_callback, reduce_lr, early_stop]
    return callbacks


def train_fit(
    subject, tested_on, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False
):
    # setup results and models folder
    folder = f"../Results/indiviuals/{model_name}/S{subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_file = f"{folder}S{subject}_{model_name}.hdf5"

    window_object = window_generator(subject)
    if tested_on == None:
        tested_on = subject

    # Get all dataset
    train_set, val_set, _ = window_object.make_dataset()
    ############################################################################
    # Load and compile new model
    K.clear_session()
    model = model_dic[model_name](window_object)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr),
                  loss=SPLoss(loss_factor))

    callbacks = model_callbacks(model_file)

    if load_best:
        try:
            model.load_weights(model_file)
        except:
            print("No saved model existing. weights will be initialized")
    ############################################################################
    # Train Model
    # Train or load the best model the model
    try:
        if not eval_only:
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=callbacks)

            plot_learning_curve(history, folder)
            plt.close()
            # Load the best model
            model.load_weights(model_file)
        else:
            history = "No training was conducted"
            model.load_weights(model_file)
    except KeyboardInterrupt:
        history = "\n\nTrains stopped manually"
        print(history)
    except OSError:  # If no saved model to be evaluated exist
        print("No saved model existing. weights will be initialized")
    ############################################################################
    # Get predictions and real values
    window_object = window_generator(tested_on)
    test_set = window_object.evaluation_set
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]

    ################ Evaluation and plot ################
    weight = subject_details[f"S{test_subject}"]["weight"]
    r2_score = nan_R2(y_true, y_pred)
    rmse_result, max_error = nan_rmse(y_true, y_pred)
    nrmse = normalized_rmse(y_true*weight, y_pred*weight)
    # Change the folder to the test subject folder after loading the model
    folder = f"../Results/indiviuals/{model_name}/S{tested_on}/"
    plot_results(y_true, y_pred, out_labels,
                 r2_score, rmse_result, max_error,
                 nrmse, folder)
    return history, y_true, y_pred, r2_score, rmse_result, nrmse


if __name__ == "__main__":
    K = keras.backend
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    else:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        gpu_index = 0
        tf.config.experimental.set_visible_devices(
            devices=gpus[gpu_index], device_type='GPU')

    tf.random.set_seed(42)

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    # Choose features and labels
    # Used EMG features
    features = ["RMS", "ZC", "WL", "AR"]

    # Used sensors
    sensors = [6, 8, 9]
    sensors = [f'sensor {x}' for x in sensors]
    # True if you want to use knee angle as an extra input
    add_knee = False
    # Labels to be predicted
    out_labels = ["ankle moment"]
    # Loss factor to prevent ankle slip
    loss_factor = 2
    # Window object parameters

    input_width = 20
    shift = 1
    label_width = 1
    batch_size = 8

    window_generator = partial(create_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width,
                               batch_size=batch_size, features=features,
                               sensors=sensors, add_knee=add_knee,
                               out_labels=out_labels)
    model_dic = {}
    # model_dic["FF model"] = create_ff_model
    # model_dic["CNN model"] = create_conv_model
    model_dic["LSTM model"] = create_lstm_model

    r2_results = pd.DataFrame(columns=model_dic.keys())
    rmse_results = pd.DataFrame(columns=model_dic.keys())
    nrmse_results = pd.DataFrame(columns=model_dic.keys())
    predictions = {}

    subjects = ["06", "08", "09"]
    for test_subject in subjects:
        for model_name in model_dic:
            history, y_true, y_pred, r2, rmse, nrmse = train_fit(
                subject=test_subject,
                tested_on=None,
                model_name=model_name,
                epochs=1000, lr=0.001,
                eval_only=False,
                load_best=False,)

            predictions[model_name] = y_pred

            r2_results.loc[f"S{test_subject}", model_name] = r2[0]
            rmse_results.loc[f"S{test_subject}", model_name] = rmse[0]
            nrmse_results.loc[f"S{test_subject}", model_name] = nrmse[0]
            # print(model_name)
            plt.close()
        plot_models(predictions, y_true, out_labels, test_subject,
                    path="../Results/indiviuals/",)
        plt.close()
    add_mean_std(r2_results)
    add_mean_std(rmse_results)
    add_mean_std(nrmse_results)
    
    r2_results.to_csv("../Results/indiviuals/R2_results.csv")
    rmse_results.to_csv("../Results/indiviuals/RMSE_results.csv")
    nrmse_results.to_csv("../Results/indiviuals/NRMSE_results.csv")
