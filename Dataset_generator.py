"""
This code will create subject's dataset. the dataset will contains:
EMG features, Joints kinematics and kinetics.
"""
import json
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def get_directories(subject: int, trials, use_DEMG=True):
    """
    Get all experiment data directories and the dataset directories
    """
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    date = subject_details[subject]["date"]
    emg_type = "DEMG" if use_DEMG else "sEMG"

    ik_path = Path("..", "OpenSim", subject, date, "IK")
    id_path = Path("..", "OpenSim", subject, date, "ID")
    periods_path = Path("..", "Outputs", subject, date, "record_periods")
    features_path = Path("..", "Outputs", subject, date, emg_type)
    output_path = Path("..", "Dataset", emg_type, subject)

    ik_files = {trial: ik_path / f"{subject}_{trial}_IK.mot" for trial in trials}
    id_files = {trial: id_path / trial / "inverse_dynamics.sto" for trial in trials}
    periods_files = {
        trial: periods_path / f"{subject}_{trial}_record_periods.csv"
        for trial in trials
    }
    features_files = {
        trial: features_path / f"{trial}_features.csv" for trial in trials
    }
    output_files = {trial: output_path / f"{trial}_dataset.csv" for trial in trials}
    return ik_files, id_files, periods_files, features_files, output_files


def load_data(ik_file: Path, id_file: Path, periods_file: Path, features_file: Path):
    """
    This function will load the trial data
    """
    IK = pd.read_csv(ik_file, header=8, sep="\t", usecols=[0, 17, 18])
    ID = pd.read_csv(id_file, header=6, sep="\t", usecols=[0, 17, 19])
    periods = pd.read_csv(periods_file, index_col=0)
    features = pd.read_csv(features_file, index_col="time")
    return IK, ID, periods, features


def merge_joints(ik_df: pd.DataFrame, id_df: pd.DataFrame, periods: pd.DataFrame):
    # Filter the moments
    id_df = lowpass_filter(id_df)
    # Merge kinematics and kinetics data on the time column
    joints_data = pd.merge(ik_df, id_df, on="time")
    moments_columns = id_df.columns[1:]
    # Start Removing Moments outside the recording periods
    # This script assumes left side only.
    left = periods[["left_start", "left_end"]].dropna() / 100
    # First remove all moment values before the start of the recording period
    condition = joints_data["time"] <= left.iloc[0, 0]
    joints_data.loc[condition, moments_columns] = np.nan
    # Loop through the recording periods rows. Remove the moment data
    # between the end of the recording period and the start
    # of the following recording period
    for i in range(1, len(left)):
        previous_end = left.loc[i - 1, "left_end"]
        current_start = left.loc[i, "left_start"]
        # Set the removing condition as stated above
        condition = joints_data["time"].between(previous_end, current_start)
        # Remove data outside the recording periods
        joints_data.loc[condition, moments_columns] = np.nan
    # By the endof the looping, the data after the last recording period's
    # end is still available, remove it as well
    condition = joints_data["time"] > left.loc[i, "left_end"]
    joints_data.loc[condition, moments_columns] = np.nan
    # Reset time to zero to match EMG
    joints_data["time"] = np.around(joints_data["time"] - joints_data["time"].min(), 3)
    return joints_data


def lowpass_filter(data: pd.DataFrame, freq=5, fs=100):
    low_pass = freq / (fs / 2)
    b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
    data.iloc[:, 1:] = filtfilt(b2, a2, data.iloc[:, 1:], axis=0)
    return data


def merge_IO(features: pd.DataFrame, joints_data: pd.DataFrame):
    """
    This function will create the dataset by merging the inputs and outputs.
    IK and ID data downsampling is hold while merging by removing time point
    that's are not shared between the features and joints data
    """
    # Merge all features and joints. Down sampling is done while when merging,
    Dataset = pd.merge(features, joints_data, on="time", how="inner")
    Dataset = Dataset.set_index("time")
    return Dataset


def get_dataset(subject: Union[str, int], use_DEMG: bool = False):
    try:
        subject = f"S{int(subject):02d}"
    except Exception as e:
        raise f"Exception {e}: Subject variable should be a number"
    # Experiment trials names
    trials = ("train_01", "train_02", "val", "test")
    # ########################## Get I/O directories ########################
    ik_files, id_files, periods_files, features_files, output_files = get_directories(
        subject, trials, use_DEMG
    )
    # ########################## Loop in each trial ###########################
    for trial in trials:
        # Load the data
        ik_df, id_df, periods_df, features_df = load_data(
            ik_files[trial],
            id_files[trial],
            periods_files[trial],
            features_files[trial],
        )
        # Merge IK, ID & record intervals together to create joint's dataset
        joints_data = merge_joints(ik_df, id_df, periods_df)
        # Merge EMG features with joints data and down sample joints data
        # to match the EMG features
        Dataset = merge_IO(features_df, joints_data)
        # Rename column and save the dataset
        new_col = {
            "knee_angle_l": "knee angle",
            "ankle_angle_l": "ankle angle",
            "ankle_angle_l_moment": "ankle moment",
            "knee_angle_l_moment": "knee moment",
        }
        Dataset = Dataset.rename(columns=new_col)
        # Save the dataset
        Dataset.to_csv(output_files[trial])


if __name__ == "__main__":
    subject = input("Please write the subject number: ")
    get_dataset(subject=subject, use_DEMG=False)
