import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tensorflow import keras

from model import model_callbacks, add_mean_std
from Custom.DataHandler import DataHandler
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def create_gm_window_generator(
    subject=None, input_width=20, shift=3, label_width=1, batch_size=64, features=["RMS"], sensors=['sensor 1'], add_knee=False, out_labels=["ankle moment"]
):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # Get subject weight.
    dataHandler = DataHandler(subject, features, sensors, add_knee, out_labels)
    # #Create Window object
    window_object = GM_WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size)
    return window_object


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

    flag = True
    for s in subject:
        if flag:
            flag = False
            window_object = window_generator(s)
            train_set = window_object.train_dataset
            val_set = window_object.val_dataset
        else:
            window_object = window_generator(s)
            temp_train = window_object.train_dataset
            temp_val = window_object.val_dataset
            
            train_set = train_set.concatenate(temp_train)
            val_set = val_set.concatenate(temp_val)
    
    train_set = GM_WindowGenerator.preprocessing(train_set, remove_nan=True,
                                                      shuffle=True, batch_size=batch_size,
                                                      drop_reminder=True,)

    val_set = GM_WindowGenerator.preprocessing(val_set, remove_nan=True,
                                                    shuffle=False,batch_size=10**6,
                                                    drop_reminder=False,)

    # Load and compile new model
    tf.keras.backend.clear_session()
    model = model_dic[model_name](window_object)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=lr), 
                  loss=SPLoss(loss_factor))

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
            
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=callbacks)
            
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
    r2_score = nan_R2(y_true, y_pred)
    rmse_result, nrmse, max_error = nan_rmse(y_true, y_pred)
    plot_results(y_true, y_pred, out_labels,
                 r2_score, rmse_result, max_error,
                 nrmse, folder)
    return history, y_true, y_pred, r2_score, rmse_result, nrmse


if __name__ == "__main__":
    tf.random.set_seed(42)

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

    # Used EMG features
    features = ["MAV", "WL"]
    # Used sensors
    sensors = [6, 7, 8, 9]
    sensors = [f'sensor {x}' for x in sensors]
    # True if you want to use knee angle as an extra input
    add_knee = False
    losses = {'ankle': 2, "knee": 1}
    # Labels to be predicted
    out_labels = ["ankle moment"]
    # Loss factor to prevent ankle slip
    loss_factor = 2
    # Window object parameters

    input_width = 20
    shift = 1
    label_width = 1
    batch_size = 8

    window_generator = partial(create_gm_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width,
                               batch_size=batch_size, features=features,
                               sensors=sensors, add_knee=add_knee,
                               out_labels=out_labels)
    model_dic = {}

    model_name = "CNN model"
    model_dic[model_name] = create_conv_model

    # Create pandas dataframe that will have all the results
    r2_results = pd.DataFrame(columns=model_dic.keys())
    rmse_results = pd.DataFrame(columns=model_dic.keys())
    nrmse_results = pd.DataFrame(columns=model_dic.keys())
    subjects = ["06", "08", "09", '10', '13', '14']

    for test_subject in subjects:
        # if test_subject not in [subjects[2*gpu_index], subjects[2*gpu_index + 1]]:
        #     continue
        train_subjects = subjects.copy()
        train_subjects.remove(test_subject)
        predictions = {}

        history, y_true, y_pred, r2, rmse, nrmse = train_fit_gm(
            subject=train_subjects,
            tested_on=test_subject,
            model_name=model_name,
            epochs=1000,
            eval_only=False,
            load_best=False)

        predictions[model_name] = y_pred
        r2_results.loc[f"S{test_subject}", model_name] = r2[0]
        rmse_results.loc[f"S{test_subject}", model_name] = rmse[0]
        nrmse_results.loc[f"S{test_subject}", model_name] = nrmse[0]

        plt.close()

        plot_models(predictions, y_true, out_labels,
                    test_subject, "../Results/GM/")
        plt.close()
    # add_mean_std(r2_results)
    # add_mean_std(rmse_results)
    # add_mean_std(nrmse_results)
    r2_results.to_csv(f"../Results/GM/R2_results{gpu_index}.csv")
    rmse_results.to_csv(f"../Results/GM/RMSE_results{gpu_index}.csv")
    nrmse_results.to_csv(f"../Results/GM/NRMSE_results{gpu_index}.csv")
