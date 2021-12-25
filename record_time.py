import pandas as pd
import numpy as np
import json

with open("subject_details.json","r") as f:
    subject_details = json.load(f)

subject = input("Please input subject number in XX format: ")
date = subject_details[f"S{subject}"]["date"]

inputs_path = f"../Data/S{subject}/{date}/record_periods/"
outputs_path = f"../Outputs/S{subject}/{date}/record_periods/"
input_files = ['train_01', 'train_02', 'val', 'test']

def interval2seq(data, side):
    for start, end in zip(periods[f"{side}_start"],periods[f"{side}_end"]):
        data.iloc[start:end]=True

left_time_sum = 0
right_time_sum = 0
for input_file in input_files:
    # Create output file name
    output_file = f"{outputs_path}{input_file}_record_periods.csv"
    # Read the data
    periods = pd.read_excel(f"{inputs_path}{input_file}.xlsx")
    # Drop last row and record time length columns
    periods.dropna(inplace=True)
    periods.drop(columns=["index", "left_time", "right_time"],inplace=True)
    # convert all data into integers
    periods = periods.astype("int16")
    # Create new boolen Dataframe
    left_df = pd.DataFrame([False]*periods.iloc[-1,1], columns=["left_side"])
    right_df = pd.DataFrame([False]*periods.iloc[-1,3], columns=["right_side"])
    # map the periods intervals to sequances
    interval2seq(left_df, "left")
    interval2seq(right_df, "right")
    # Create time column
    time = pd.DataFrame([round(i/100,4) for i in range(len(left_df))], columns=["time"])
    # Merge all data
    df = pd.concat([time,left_df,right_df], axis=1)
    # save to csv
    df.to_csv(output_file, index=False)
    # print results
    print(f"{input_file}: left time: {np.sum(left_df)[0]/100}s \t right time: {np.sum(right_df)[0]/100}s \n record_time: {len(right_df)/100}s")