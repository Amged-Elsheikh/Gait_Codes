import pandas as pd
import Force_convert_functions as grf
import json
from scipy.signal import butter, filtfilt
import re

def lowpass_filter(data):
    f = 10  # Filter frequency
    fs = 100 # Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    return filtfilt(b2, a2, data, axis=0)


def load_grf(subject, trial, side="L"):
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        
    date = subject_details[f"S{subject}"]["date"]
    grf_paths = f"../Data/S{subject}/{date}/Dynamics/Force_Data/S{subject}_{trial}_forceplate_"
    if side == "L":
        grf_paths = f"{grf_paths}1.csv"
    if side == "R":
        grf_paths = f"{grf_paths}2.csv"
    grf_data = pd.read_csv(grf_paths, header=31)
    grf_data = grf.remove_system_gap(grf_data)
    # Downsample to 100 Hz if forceplate work on 1KHz
    grf_data = grf.trial_period(grf_data, subject, trial)
    fps = subject_details[f"S{subject}"]['FP_sampling_rate']
    if fps == 1000:
        grf_data = grf_data.loc[grf_data[' DeviceFrame']%10==0]
        grf_data.reset_index(inplace=True, drop=True)
    return grf_data


def get_heel_starting_index(input_path):
    """
    Input: input data path. Type: string
    """
    # Getting Markers labels
    Markers_Label = pd.read_csv(
        input_path, header=2, nrows=0).columns.values[2:]
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Do not use set because we do not want to change the order of markers
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    # Remove asset name if any
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    # get heel index
    left_heel = unique_labels.index("L.Heel")*3 + 2
    right_heel = unique_labels.index("R.Heel")*3 + 2
    left_heel_y_column = left_heel + 1
    right_heel_y_column = right_heel + 1
    
    
    return left_heel_y_column, right_heel_y_column


def load_heels_data(subject, trial):
    if subject == None:
        subject = input("insert subject number: ")
        
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        
    date = subject_details[f"S{subject}"]["date"]
    motive_path = f"../Data/S{subject}/{date}/Dynamics/motion_data/S{subject}_{trial}.csv"
    left, right = get_heel_starting_index(motive_path)

    Markers = pd.read_csv(motive_path, header=5)
    return Markers.iloc[:, [left, right]]


def get_toe_off(start: int, data: pd.DataFrame, safety_factor=20):
    start -= safety_factor
    while data.iloc[start-1] >= data.iloc[start]:
        start -= 1
    return start


def get_heel_strike(end: int, data: pd.DataFrame, safety_factor=20):
    end += safety_factor
    while data.iloc[end+1] <= data.iloc[end]:
        end += 1
    return end


def get_periods(subject=None, trilas=["train_01", "train_02", "val", "test"]):
    if subject == None:
        subject = input("Enter subject number in XX format\n")
    subject = f"{int(subject):02d}"

        
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)[f"S{subject}"]
        
    date = subject_details["date"]
    sides = subject_details["sides"]

    for trial in trilas:
        output_dir = f"../Outputs/S{subject}/{date}/record_periods/S{subject}_{trial}_record_periods.csv"
        motive = load_heels_data(subject, trial)
        motive.columns = ['L.Heel', "R.Heel"]
        # Smooth the Mocap signal
        motive.loc[:,['L.Heel', "R.Heel"]] = lowpass_filter(motive.loc[:,['L.Heel', "R.Heel"]])
        left_periods = pd.DataFrame(
            columns=['left_start', 'left_end', 'left_time'])
        right_periods = pd.DataFrame(
            columns=['right_start', 'right_end', 'right_time'])

        data_dict = {"L": left_periods, "R": right_periods}
        for side in sides:
            grf_data = load_grf(subject, trial, side)
            # Periods unit is Mocap Frame
            stance = grf.grf_periods(grf_data)
            for period in stance:
                try:
                    # Take some extra frames when getting T.O
                    start = get_toe_off(period.start, motive.loc[:, f"{side}.Heel"]) - 5
                    # Take less frames when getting H.S
                    end = get_heel_strike(period.stop, motive.loc[:, f"{side}.Heel"]) - 5
                    data_dict[side].loc[len(data_dict[side])] = [start, end, end-start]
                except:
                    pass
            
        periods = pd.concat([left_periods, right_periods])
        periods.to_csv(output_dir)


if __name__ == '__main__':
    subject=None
    if subject:
        subject = f"{int(subject):02d}"
    trilas=["train_01", "train_02", "val", "test"]
    get_periods(subject, trilas)