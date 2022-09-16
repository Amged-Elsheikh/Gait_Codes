import json
from typing import *

import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from intrasubject_training import *
from Custom.PlottingFunctions import *
from Custom.TFModels import *
from Custom.TFModelEvaluation import *
from Custom.OneSideWindowGenerator import *


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
        
        r2, rmse, nrmse, y_true, y_pred, models_compare_pdf, models_compare_svg=\
            train_fit_gm(subject=train_subjects,
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

