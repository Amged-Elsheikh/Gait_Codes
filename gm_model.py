import json
import os
from typing import *

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tensorflow import keras

from model import model_callbacks, add_mean_std
from intrasubject_training import *
from Custom.DataHandler import DataHandler
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def create_gm_window_generator(subject=None, input_width=20, shift=3, 
                                label_width=1, batch_size=64, features=["RMS"],
                                sensors=['sensor 1'], add_knee=False, 
                                out_labels=["ankle moment"], emg_type='sEMG'):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # Get subject weight.
    dataHandler = DataHandler(subject, features, sensors, add_knee, out_labels, emg_type)
    # #Create Window object
    window_object = GM_WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size)
    return window_object


def train_fit_gm(subject: List[str], tested_on: str, model_name: str, models_dic: Dict,
              epochs=1, lr=0.001, eval_only=False,
              load_best=False, joint='ankle', input_width=20,
              shift=1, label_width=1, batch_size=8,
              features=['RMS'], sensors=['sensor 1'], add_knee=False,
              out_labels=[f"ankle moment"], emg_type='DEMG'):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
   ################################## Get Files ##################################
    model_file,\
        models_compare_pdf, models_compare_svg,\
        predictions_pdf, predictions_svg,\
        learning_curve_pdf, learning_curve_svg = get_files(features, sensors,
                                                           joint, tested_on, 
                                                           model_name, emg_type,
                                                           is_general_model=True)

    ######### Create Window Object and load trainining and Validation sets ########
    window_generator = partial(create_gm_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width, batch_size=batch_size,
                               features=features, sensors=sensors,
                               add_knee=add_knee, out_labels=out_labels, emg_type=emg_type)
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

    ####################### Train or Load Model ###################################
    loss_factor = LOSSES[joint]
    model = compile_and_train(model_name, model_file, models_dic,
                              train_set, val_set, loss_factor,
                              learning_curve_pdf, learning_curve_svg,
                              window_object, epochs,
                              lr, eval_only, load_best)

    ######################### Make Estimations #####################################
    if tested_on == None:
        tested_on = subject
    y_true, y_pred = get_estimations(tested_on, window_generator, model)

    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation_and_plot(y_true, y_pred,
                                                       tested_on, out_labels,
                                                       predictions_pdf, predictions_svg)
    return y_true, y_pred, r2_score, rmse_result, nrmse, models_compare_pdf, models_compare_svg


if __name__ == "__main__":
    GPU_num = 0
    tf.random.set_seed(42)
    joint = 'ankle'
    emg_type='DEMG'

    select_GPU(GPU_num)
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
    # Labels to be predicted
    out_labels = [f"{joint} moment"]
    models_dic = {}

    model_name = "RCNN model"
    models_dic[model_name] = create_conv_model

    # Create pandas dataframe that will have all the results
    results = pd.read_csv('../Results/GM_Results.csv', index_col=0)
    subjects = ["06", "08", "09", '10', '13', '14', '16']

    for test_subject in subjects:
        if test_subject != '16':
            continue
        train_subjects = subjects.copy()
        train_subjects.remove(test_subject)
        
        y_true, y_pred, r2,\
             rmse, nrmse, models_compare_pdf\
                , models_compare_svg = train_fit_gm(
            subject=train_subjects,
            tested_on=test_subject,
            model_name=model_name,
            models_dic=models_dic,
            epochs=1000,
            eval_only=True,
            load_best=False,
            joint=joint,
            input_width=20,
            shift=1,
            label_width=1,
            batch_size=8,
            features=features,
            sensors=sensors,
            add_knee=add_knee,
            out_labels=out_labels,
            emg_type=emg_type)

        print(f"R2: {r2[0]*100}")
        print(f'RMSE :{rmse[0]}')
        print(f'NRMSE :{nrmse[0]}')

        results.loc[int(test_subject), 'R2'] = r2[0]
        results.loc[int(test_subject), 'RMSE'] = rmse[0]
        results.loc[int(test_subject), 'NRMSE'] = nrmse[0]
        results.to_csv(f"../Results/GM_{emg_type}_{model_name} GM_Results.csv")

