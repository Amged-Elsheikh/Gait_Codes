from typing import *
import pandas as pd
from intrasubject_training import *


if __name__ == "__main__":
    tf.random.set_seed(42) # Select the random number generator to ensure reproducibility of the results
    # Select the GPU to be used
    GPU_num = 0
    select_GPU(GPU_num)
    # Select the subjectt for training/testing
    test_subjects = [6, 8, 9, 10, 13, 14, 16]
    ############ Models names functions ############
    # If you create new model, just give it a name and pass the function to the dictionary
    models_dic = {}
    models_dic['MLP'] = create_ff_model
    models_dic['LSTM'] = create_lstm_model
    models_dic['RCNN'] = create_conv_model
    
    ################################################
    joint = 'ankle' # Joint that data will be predicted (knee or ankle)
    emg_type = 'DEMG' # EMG features type (sEMG or DEMG)
    add_knee = False # True if you want to use knee angle as an extra input
    ############ iteration options ############
    ### If you want to iterate through different variable, uncomment it and do not forget to handle indentation properly.
    # for emg_type in ['sEMG','DEMG']: # Iterate through theemg_type
    # for joint in ['knee', 'ankle']: # Iterate through the output labels
    for model_name in models_dic.keys():
        if type(joint) == str:
            out_labels = [f"{joint} moment"]
        # In case you want to train a model to estimate both knee and ankle (Current codes can't handle the situation yet)
        elif type(joint) == list:
            out_labels = [f'{i} moment' for i in joint]
            raise "You are trying to train a model for knee and ankle moment estimation.\
                Current codes can't handle the situation yet"
        ### Load results file. Please ensure the format and directory
        csv_file = f"{joint} features evaluation.csv"
        # Load the whole dataset
        eval_df = pd.read_csv(f'../Results/{csv_file}',
                            index_col=[0, 1], header=[0, 1])
        # Get the sensors from the df
        sensors_list: List[str] = []
        for col in eval_df.droplevel(1, 1).columns:
            if col not in sensors_list:
                sensors_list.append(col)
        # Loop through subjects and features
        for subject, features_string in eval_df.index:
            # train/test selected subjects only
            if subject not in test_subjects:
                continue
            # Format the subject name
            test_subject = f"{subject:02d}"
            # convert features_string to a list
            features: List[str] = features_string.split("+")
            # loop through sensors
            for sensors_id in sensors_list:
                sensors_num = sensors_id.split("+")
                # Create sensors list for the DataHandling method
                sensors = [f'sensor {x}' for x in sensors_num]
                r2, rmse, nrmse = train_fit(subject=test_subject, # the subject to train the model on
                                            tested_on=None, # Subject number, that the model will be evaluated on
                                            models_dic=models_dic, # Dictionary of all models functions
                                            model_name=model_name, # Name of the model to be used from the models_dic
                                            epochs=1000, # Maximum number of epochs to train
                                            lr=0.001, # learning rate
                                            eval_only=True, # Do you want to evaluate only (no training). Will load the best model if it exists
                                            load_best=True, # When training new model, do you want to start from a saved models
                                            joint=joint, # joint to be predicted
                                            input_width=20, # the length of the input time series
                                            shift=1, # Output time point distance from thelast input's point on the time series
                                            label_width=1, # How many points you want to predict (set to 1 for now)
                                            batch_size=8, # The batch size
                                            features=features, # What features you want to use
                                            sensors=sensors, # What are the sensors you want to use
                                            add_knee=add_knee, # Do you want to use knee angle as an input
                                            out_labels=out_labels, # Output labels
                                            emg_type=emg_type) # Do you want to use 'sEMG' or 'DEMG'
                # Add evvaluation metrices to the Results file
                eval_df.loc[(int(test_subject), features_string),
                            (sensors_id, "R2")] = r2[0]
                eval_df.loc[(int(test_subject), features_string),
                            (sensors_id, "RMSE")] = rmse[0]
                eval_df.loc[(int(test_subject), features_string),
                            (sensors_id, "NRMSE")] = nrmse[0]
                # Save after addin new data (training can take weeks, so better to keep saving the results)
                eval_df.to_csv(f"../Results/Results_{emg_type} {model_name} {csv_file}")
                plt.close()
            plt.close()
