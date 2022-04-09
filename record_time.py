import pandas as pd
import numpy as np
import json

with open("subject_details.json", "r") as f:
    subject_details = json.load(f)

subject = input("Please input subject number in XX format: ")
date = subject_details[f"S{subject}"]["date"]

inputs_path = f"../Data/S{subject}/{date}/record_periods/"
outputs_path = f"../Outputs/S{subject}/{date}/record_periods/"
trials = ['train_01', 'train_02', 'val', 'test']


def interval2seq(data, side):
    for start, end in\
        zip(periods[f"{side}_start"][:-1],
            periods[f"{side}_end"][:-1]):

        data.iloc[start:end] = True


for trial in trials:
    # Create output file name
    output_file = f"{outputs_path}{trial}_record_periods.csv"
    # Read the data
    periods = pd.read_excel(f"{inputs_path}{trial}.xlsx")
    # Drop last row and record time length columns
    periods.dropna(inplace=True)
    periods.drop(columns=["index", "left_time"], inplace=True)
    # convert all data into integers
    periods = periods.astype("int16")
    # Create new boolen Dataframe
    left_df = pd.DataFrame([False]*periods.iloc[-1, 1], columns=["left_side"])
    # map the periods intervals to sequances
    interval2seq(left_df, "left")
    # Create time column
    time = pd.DataFrame([round(i/100, 4)
                        for i in range(len(left_df))], columns=["time"])
    # Merge all data
    df = pd.concat([time, left_df], axis=1)
    # save to csv
    df.to_csv(output_file, index=False)
    # print results
    print(f"{trial}: left time: {np.sum(left_df)[0]/100}s \t")
