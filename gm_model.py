import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tensorflow import keras

from model import create_window_generator, model_callbacks, add_mean_std
from Custom.DataHandler import DataHandler
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def train_fit_gm(subject, tested_on, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False,):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
    # #Create Results and model folder
    folder = f"../Results/GM/{model_name}/S{tested_on}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_file = f"{folder}S{tested_on}_{model_name}.hdf5"
    # Make dataset
    # #FOR NOW I HAVE 2 SUBJECTS FOR TRAINING, I'LL GENERALIZE THE CODE TO ACCEPT MORE SUBJECTS LATER
    window_object_1 = window_generator(subject[0])
    window_object_2 = window_generator(subject[1])
    # Get all dataset
    train_set_1, val_set_1 = window_object_1.get_gm_train_val_dataset()
    train_set_2, val_set_2 = window_object_2.get_gm_train_val_dataset()
    train_set = window_object_1.preprocessing(
        train_set_1.concatenate(train_set_2),
        remove_nan=True, shuffle=True,
        batch_size=None, drop_reminder=True,)
    
    val_set = window_object_1.preprocessing(
        val_set_1.concatenate(val_set_2),
        remove_nan=True, shuffle=False,
        batch_size=None, drop_reminder=False,)
    
    # Load and compile new model
    tf.keras.backend.clear_session()
    model = model_dic[model_name](window_object_1)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr), loss=SPLoss(loss_factor)
    )
    
    callbacks = model_callbacks(model_file)

    if load_best or eval_only:
        try:
            model.load_weights(model_file)
        except:
            print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Train Model
    try:  # Train or load the best model the model
        if not eval_only:
            history = model.fit(
                x=train_set, validation_data=val_set,
                epochs=epochs, callbacks=callbacks,
            )
            plot_learning_curve(history, folder)
            plt.close()
        else:
            history = "No training was conducted"
    except KeyboardInterrupt:
        history = "\n\nTrains stopped manually"
        print(history)
    except OSError:  # If no saved model to be evaluated exist
        print("No saved model existing. weights will be initialized")
    ############################################################################
    # Get predictions and real values
    test_window = window_generator(tested_on)
    test_set = test_window.evaluation_set
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]

    ################ Evaluation and plot ################
    weight = subject_details[f"S{tested_on}"]["weight"]
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
    tf.random.set_seed(42)
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    gpu_index = 0
    tf.config.experimental.set_visible_devices(
        devices=gpus[gpu_index], device_type="GPU")
    # Check for GPU
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    # Get all subjects details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Choose features and labels
    # Used EMG features
    features = ["RMS", "ZC", "WL", "AR"]

    # Used sensors
    sensors = [6, 7, 8, 9]
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

    model_dic["FF model"] = create_ff_model
    model_dic["CNN model"] = create_conv_model
    model_dic["LSTM model"] = create_lstm_model

    # Create pandas dataframe that will have all the results
    r2_results = pd.DataFrame(columns=model_dic.keys())
    rmse_results = pd.DataFrame(columns=model_dic.keys())
    nrmse_results = pd.DataFrame(columns=model_dic.keys())
    subjects = ["06", "08", "09"]

    for test_subject in subjects:
        train_subjects = subjects.copy()
        train_subjects.remove(test_subject)
        predictions = {}

        for model_name in model_dic.keys():
            print(model_name)

            history, y_true, y_pred, r2, rmse, nrmse = train_fit_gm(
                subject=train_subjects,
                tested_on=test_subject,
                model_name=model_name,
                epochs=500,
                eval_only=True,
                load_best=False)
            
            predictions[model_name] = y_pred
            r2_results.loc[f"S{test_subject}", model_name] = r2[0]
            rmse_results.loc[f"S{test_subject}", model_name] = rmse[0]
            nrmse_results.loc[f"S{test_subject}", model_name] = nrmse[0]
            
            plt.close()

        plot_models(predictions, y_true, path="../Results/GM/",
                    subject=test_subject)
        plt.close()
    add_mean_std(r2_results)
    add_mean_std(rmse_results)
    add_mean_std(nrmse_results)
    r2_results.to_csv("../Results/GM/R2_results.csv")
    rmse_results.to_csv("../Results/GM/RMSE_results.csv")
    nrmse_results.to_csv("../Results/GM/NRMSE_results.csv")
