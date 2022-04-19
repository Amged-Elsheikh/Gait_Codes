import pandas as pd
import Force_convert_functions as grf
import json
import re

with open("subject_details.json", "r") as f:
    subject_details = json.load(f)


def load_grf(subject, trial, side="L"):
    date = subject_details[f"S{subject}"]["date"]
    grf_paths = f"../Data/S{subject}/{date}/Dynamics/Force_Data/S{subject}_{trial}_forceplate_"
    if side == "L":
        grf_paths = f"{grf_paths}1.csv"
    if side == "R":
        grf_paths = f"{grf_paths}2.csv"
    grf_data = pd.read_csv(grf_paths, header=31)
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
    return end-3


def get_periods(subject=None, trilas=["train_01", "train_02", "val", "test"], sides=["L", "R"]):
    if subject == None:
        subject = input("Enter subject number in XX format\n")
    date = subject_details[f"S{subject}"]["date"]
    for trial in trilas:
        output_dir = f"../Outputs/S{subject}/{date}/record_periods/{trial}_record_periods.csv"
        motive = load_heels_data(subject, trial)
        motive.columns = ['L.Heel', "R.Heel"]
        left_periods = pd.DataFrame(
            columns=['left_start', 'left_end', 'left_time'])
        right_periods = pd.DataFrame(
            columns=['right_start', 'right_end', 'left_time'])
        for side in sides:
            grf_data = load_grf(subject, trial, side)
            stance = grf.grf_periods(grf_data)
            if side == 'L':
                for period in stance[:-1]:
                    start = get_toe_off(period.start, motive.loc[:, "L.Heel"])
                    end = get_heel_strike(period.stop, motive.loc[:, "L.Heel"])
                    left_periods.loc[len(left_periods)] = [
                        start, end, end-start]
            elif side == 'R':
                for period in stance[:-1]:
                    start = get_toe_off(period.start, motive.loc[:, "R.Heel"])
                    end = get_heel_strike(period.stop, motive.loc[:, "R.Heel"])
                    right_periods.loc[len(right_periods)] = [
                        start, end, end-start]
        periods = pd.concat([left_periods, right_periods])
        periods.to_csv(output_dir)


if __name__ == '__main__':
    get_periods("06", sides="L")
