"""
This code will create subject's dataset. the dataset will contains: EMG features, Joints kinematics and kinetics.
"""
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import json
from typing import *


def get_directories(subject: int, trials, use_DEMG=True) -> List[List[str]]:
    '''
    Load all experiment data directories and the dataset directories
    '''
    # Load the subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        date = subject_details[f"S{subject}"]["date"]
    # Get IK directories
    ik_path = f"../OpenSim/S{subject}/{date}/IK/"
    IK_files = list(map(lambda x: f"{ik_path}S{subject}_{x}_IK.mot", trials))
    # Get ID directories
    id_path = f"../OpenSim/S{subject}/{date}/ID/"
    ID_files = list(
        map(lambda x: f"{id_path}{x}/inverse_dynamics.sto", trials))
    # Get the record's periods directories
    record_periods_path = f"../Outputs/S{subject}/{date}/record_periods/"
    periods_files = list(
        map(lambda x: f"{record_periods_path}S{subject}_{x}_record_periods.csv", trials))
    # Get the EMG features directories
    features_path = f"../Outputs/S{subject}/{date}/EMG/"
    Features_files = list(
        map(lambda x: f"{features_path}{x}_features.csv", trials))
    # Get Dataset directories
    if use_DEMG:
        emg_type = 'DEMG'
    else:
        emg_type = 'sEMG'
    output_folder = f"../Dataset/{emg_type}/S{subject}/"
    output_files = list(
        map(lambda x: f"{output_folder}{x}_dataset.csv", trials))
    return [IK_files, ID_files, periods_files, Features_files, output_files]


def load_data(ik_file: str, id_file: str, periods_file: str, features_file: str) -> List[pd.DataFrame]:
    '''
    This function will load the trial data
    '''
    # Load IK data
    IK = pd.read_csv(ik_file, header=8, sep='\t', usecols=[0, 17, 18])
    # Load ID data
    ID = pd.read_csv(id_file, header=6, sep='\t', usecols=[0, 17, 19])
    # Load record interval data
    periods = pd.read_csv(periods_file, index_col=0)
    # Load EMG features
    features = pd.read_csv(features_file, index_col='time')
    return [IK, ID, periods, features]


def merge_joints(IK: pd.DataFrame, ID: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Join kinematics and kinetics data into a single pandas dataframe along with record periods
    dataframe which used to remove the moment values (set as NaN) outside the recording period (space)
    """
    # Filter the moments
    ID.iloc[:, :] = lowpass_filter(ID)
    # Merge kinematics and kinetics data on the time column
    joints_data = pd.merge(IK, ID, on='time', how='inner')
    # work on left side
    moments_columns = ID.columns[1:]
    ######################## Start Removing Moments outside the recording periods ########################
    # This script assumes left side only. Use dropna to remove the last row (shows how long the recording period was)
    left = periods[['left_start', 'left_end']].dropna()
    # First remove all moment values before the staqrt of the recording period
    condition = joints_data['time'] <= left.iloc[0, 0]/100
    joints_data.loc[condition, moments_columns] = np.nan
    # Loop through the recording periods rows. Remove the moment data between the end of the recording period 
    # and the start of the following recording period
    for i in range(1, len(left)):
        # divide by 100 to get the time
        previous_end = left.loc[i-1, 'left_end']/100
        current_start = left.loc[i, 'left_start']/100
        # Set the removing condition as stated above
        condition = (previous_end <= joints_data['time']) & (
            joints_data['time'] <= current_start)
        # Remove data outside the recording periods
        joints_data.loc[condition, moments_columns] = np.nan
    # By the endof the looping, the data after the last recording period's end is still available, remove it as well
    condition = joints_data['time'] > left.loc[i, 'left_end']/100
    joints_data.loc[condition, moments_columns] = np.nan
    # Reset time to zero to match EMG
    joints_data = reset_time(joints_data)
    return joints_data


def lowpass_filter(data: pd.DataFrame, freq=5, fs=100):
    low_pass = freq/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    data.iloc[:, 1:] = filtfilt(b2, a2, data.iloc[:, 1:], axis=0)
    return data


def reset_time(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function will reset the time from zero by removing the minimum value
    '''
    # start_time = data['time'].min()
    # data['time'] = data['time'].apply(lambda x: x-start_time)
    data['time'] = data['time'] - data['time'].min()
    # Round the time into 3 digits (very important because of python limitations)
    data['time'] = np.around(data['time'], 3)
    return data


def merge_IO(features: pd.DataFrame, joints_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will create the dataset by merging the inputs and outputs.
    IK and ID data downsampling is hold while merging by removing time point
    that's are not shared between the features and joints data
    """
    # Merge all features and joints. Down sampling is done while merging using 'inner' argument
    Dataset = pd.merge(left=features, right=joints_data,
                       on='time', how='inner')
    Dataset.set_index("time", inplace=True)
    return Dataset


def get_dataset(subject: Union[str, int], use_DEMG: bool = False):
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    # Experiment trials names
    trials = ('train_01', 'train_02', 'val', 'test')
    ########################### Get I/O directories ###########################
    IK_files, ID_files, periods_files, Features_files, output_files = get_directories(
        subject, trials, use_DEMG)
    ########################### Loop in each trial ###########################
    for trial in range(len(trials)):
        # Load the data
        IK, ID, periods, features = load_data(IK_files[trial], ID_files[trial],
                                              periods_files[trial], Features_files[trial])
        # Merge IK, ID & record intervals together to create joint's dataset
        joints_data = merge_joints(IK, ID)
        # Merge EMG features with joints data and down sample joints data to match the EMG features
        joints_data = select_walking_trials(joints_data, periods)
        Dataset = merge_IO(features, joints_data)
        # Rename column and save the dataset
        new_col = {'knee_angle_l': "knee angle",
                   'ankle_angle_l': 'ankle angle',
                   'ankle_angle_l_moment': "ankle moment",
                   'knee_angle_l_moment': "knee moment"}
        Dataset.rename(columns=new_col, inplace=True)
        # Save the dataset
        Dataset.to_csv(output_files[trial])


if __name__ == "__main__":
    subject = input('Please write the subject number: ')
    get_dataset(subject=subject, use_DEMG=False)
