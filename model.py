import json

import pandas as pd
import tensorflow as tf
from modelTrainingFunc import train_fit
from utilities.TFModels import create_conv_model, select_GPU


tf.random.set_seed(42)


if __name__ == "__main__":
    # Get all subjects details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    select_GPU(0)
    emg_type = "sEMG"
    features_list = [
        ["WL"],
        ["MAV", "WL"],
    ]
    sensors_list = [[6, 7, 8, 9], [6, 8, 9]]
    model_name = "RCNN"
    models_dic = {model_name: create_conv_model}
    subjects = ["06", "08", "09", "10", "13", "14", "16"]
    for sensors, features in zip(sensors_list, features_list):
        results = pd.DataFrame()
        for test_subject in subjects:
            train_subjects = [s for s in subjects if s != test_subject]
            y_true, y_pred, r2, rmse, nrmse = train_fit(
                subjects=train_subjects,
                tested_on=test_subject,
                model_name=model_name,
                models_dict=models_dic,
                features=features,
                sensors=[f"sensor {x}" for x in sensors],
                emg_type="sEMG",
                joint="ankle",
                is_general_model=True,
                eval_only=True,
                load_best=True,
                epochs=1000,
                input_width=20,
                shift=1,
                label_width=1,
                batch_size=8,
            )
            print(f"R2: {r2[0]}")
            print(f"RMSE :{rmse[0]}")
            print(f"NRMSE :{nrmse[0]}")
            results.loc[int(test_subject), "R2"] = r2[0]
            results.loc[int(test_subject), "RMSE"] = rmse[0]
            results.loc[int(test_subject), "NRMSE"] = nrmse[0]
            results.to_csv(
                f"../Results/GM_{emg_type}_{model_name}"
                f"{'+'.join(features)} GM_Results.csv"
            )
