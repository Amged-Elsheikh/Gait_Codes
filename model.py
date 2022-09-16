import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams

from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *
from intrasubject_training import *


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
plt.style.use("ggplot")


if __name__ == "__main__":
    tf.random.set_seed(42) # Select the random number generator to ensure reproducibility of the results
    select_GPU(0) # Select the GPU to be used
    subject = 6 # Select the subjectt for training
    tested_on = None # Select the subjectt for testing. if none will test on train subject
    subject = f"{subject:02d}"
    ######################### Model I/O #########################
    add_knee = False # True if you want to use knee angle as an extra input
    features = ["MAV", "WL"] # Select features
    emg_type = 'DEMG' # EMG features type (sEMG or DEMG)
    sensors_num = [6, 7, 8, 9] # select sensors numbers (1~9)
    sensors = [f'sensor {x}' for x in sensors_num]
    joint = 'ankle' # Joint that data will be predicted (knee or ankle)
    out_labels = [f'{joint} moment',]
    ###################### Window Parameters ######################
    input_width = 20
    shift = 1
    label_width = 1
    batch_size = 8
    #################### Models names functions ####################
    # If you create new model, just give it a name and pass the function to the dictionary
    models_dic = {}
    models_dic['MLP'] = create_ff_model
    models_dic['LSTM'] = create_lstm_model
    models_dic['RCNN'] = create_conv_model
    ############ Create pd dataframe to hold the results ############
    r2_results = pd.DataFrame(columns=models_dic.keys())
    rmse_results = pd.DataFrame(columns=models_dic.keys())
    nrmse_results = pd.DataFrame(columns=models_dic.keys())
    predictions = {}
    ################################################
    for model_name in models_dic:
        r2, rmse, nrmse, y_true, y_pred, models_compare_pdf, models_compare_svg =\
            train_fit(subject=subject, # the subject to train the model on
            tested_on=tested_on, # Subject number, that the model will be evaluated on
            models_dic=models_dic, # Dictionary of all models functions
            model_name=model_name, # Name of the model to be used from the models_dic
            epochs=1000, # Maximum number of epochs to train
            lr=0.001, # learning rate
            eval_only=True, # Do you want to evaluate only (no training). Will load the best model if it exists
            load_best=True, # When training new model, do you want to start from a saved models
            joint=joint, # joint to be predicted
            input_width=input_width, # the length of the input time series
            shift=shift, # Output time point distance from thelast input's point on the time series
            label_width=label_width, # How many points you want to predict (set to 1 for now)
            batch_size=batch_size, # The batch size
            features=features, # What features you want to use
            sensors=sensors_num, # What are the sensors you want to use
            add_knee=add_knee, # Do you want to use knee angle as an input
            out_labels=out_labels, # Output labels
            emg_type=emg_type) # Do you want to use 'sEMG' or 'DEMG'
        
        predictions[model_name] = y_pred
        r2_results.loc[f"S{subject}", model_name] = r2[0]
        rmse_results.loc[f"S{subject}", model_name] = rmse[0]
        nrmse_results.loc[f"S{subject}", model_name] = nrmse[0]
        plt.close()
    plot_models(predictions, y_true, out_labels, subject,
                folders=[models_compare_pdf, models_compare_svg],)
    plt.close()
    add_mean_std(r2_results)
    add_mean_std(rmse_results)
    add_mean_std(nrmse_results)

    r2_results.to_csv("../Results/indiviuals/R2_results.csv")
    rmse_results.to_csv("../Results/indiviuals/RMSE_results.csv")
    nrmse_results.to_csv("../Results/indiviuals/NRMSE_results.csv")

