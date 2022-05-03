"""This code will create each subject dataset. the dataset will contains on EMG features, Joints kinematics and kinetics.
"""
import pandas as pd
import numpy as np
import json


def get_directories(subject, trials):
    # Load the subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        date = subject_details[f"S{subject}"]["date"]

    ik_path = f"../OpenSim/S{subject}/{date}/IK/"
    IK_files = list(map(lambda x: f"{ik_path}S{subject}_{x}_IK.mot", trials))

    id_path = f"../OpenSim/S{subject}/{date}/ID/"
    ID_files = list(
        map(lambda x: f"{id_path}{x}/inverse_dynamics.sto", trials))

    record_periods_path = f"../Outputs/S{subject}/{date}/record_periods/"
    periods_files = list(
        map(lambda x: f"{record_periods_path}S{subject}_{x}_record_periods.csv", trials))

    features_path = f"../Outputs/S{subject}/{date}/EMG/"
    Features_files = list(
        map(lambda x: f"{features_path}{x}_features.csv", trials))

    output_folder = f"../Dataset/S{subject}/"
    output_files = list(
        map(lambda x: f"{output_folder}{x}_dataset.csv", trials))

    return IK_files, ID_files, periods_files, Features_files, output_files


def load_data(ik_file, id_file, periods_file, features_file):
    # Load IK data
    IK = pd.read_csv(ik_file, header=8, sep='\t', usecols=[0, 17, 18])
    # Load ID data
    ID = pd.read_csv(id_file, header=6, sep='\t', usecols=[0, 17, 19])
    # Load record interval data
    periods = pd.read_csv(periods_file, index_col=0)
    
    # Load EMG features
    features = pd.read_csv(features_file, index_col='time')
    return IK, ID, periods, features


def merge_joints(IK: pd.DataFrame, ID: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Join kinematics and kinetics data into a single pandas dataframe along with record periods dataframe which used to filter the dataset by removing all periods where ID solution was not available (no GRF data)
    """
    # Merge kinematics and kinetics data
    joints_data = pd.merge(IK, ID, on='time', how='inner')

    left = periods[['left_start', 'left_end']].dropna()
    # work on left side
    left_joints = ['knee_angle_l_moment', 'ankle_angle_l_moment']
    condition = joints_data['time']<=left.iloc[0, 0]/100
    joints_data.loc[condition, left_joints] = np.nan
    for i in range(1, len(left)):
        # divide by 100 to get the time
        previous_end = left.loc[i-1, 'left_end']/100
        current_start = left.loc[i, 'left_start']/100
        condition = (previous_end<=joints_data['time']) & (joints_data['time']<=current_start)
        joints_data.loc[condition, left_joints] = np.nan
    condition = joints_data['time'] > left.loc[i, 'left_end']/100
    joints_data.loc[condition, left_joints] = np.nan
    # Reset time to zero to match EMG
    joints_data = reset_time(joints_data)
    return joints_data


def reset_time(data: pd.DataFrame) -> pd.DataFrame:
    start_time = data['time'].min()
    data['time'] = data['time'].apply(lambda x: x-start_time)
    data['time'] = np.around(data['time'], 3)
    return data


def merge_IO(features: pd.DataFrame, joints_data: pd.DataFrame) -> pd.DataFrame:
    """
    downsampling is hold while merging by removing points
    """
    # Merge all features and joints. Down sampling is done while merging
    Dataset = pd.merge(left=features, right=joints_data,
                       on='time', how='inner')
    Dataset.set_index("time", inplace=True)
    return Dataset


def get_dataset(subject=None) -> None:
    # If subject number wasn't provided ask the user to manually input it
    if subject == None:
        subject = input("Please write subject number in a format XX: ")
    # Experiment trials names
    trials = ('train_01', 'train_02', 'val', 'test')
    ########################### Get I/O directories ###########################
    IK_files, ID_files, periods_files, Features_files, output_files = get_directories(
        subject, trials)
    ########################### Loop in each trial ###########################
    for ik_file, id_file, periods_file, features_file, output_name\
            in zip(IK_files, ID_files, periods_files, Features_files, output_files):

        IK, ID, periods, features = load_data(
            ik_file, id_file, periods_file, features_file)
        # Merge IK, ID & record intervals together to create joint's dataset
        joints_data = merge_joints(IK, ID, periods)
        # Merge EMG features with joints data and down sample joints data to match the EMG features
        Dataset = merge_IO(features, joints_data)
        # Rename column and save the dataset
        new_col = {'knee_angle_l': "knee angle",  # Test as input
                   'ankle_angle_l': 'ankle angle',  # will not be used
                   'ankle_angle_l_moment': "ankle moment",
                   'knee_angle_l_moment': "knee moment"}
        Dataset.rename(columns=new_col, inplace=True)
        Dataset.to_csv(output_name)


if __name__ == "__main__":
    get_dataset()
