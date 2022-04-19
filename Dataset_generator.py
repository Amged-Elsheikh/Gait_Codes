"""This code will create each subject dataset. the dataset will contains on EMG features, Joints kinematics and kinetics.
"""
import pandas as pd
import numpy as np
import json


def load_IK(ik_file: str) -> pd.DataFrame:
    """
    Load knee and ankle joints angles
    ik_file (str): full directory name
    """
    IK = pd.read_csv(ik_file, header=8, sep='\t', usecols=[0, 10, 11, 17, 18])
    return IK


def load_ID(id_file: str) -> pd.DataFrame:
    """
    load knee and ankle joints moments
    id_file (str): full directory name
    """
    ID = pd.read_csv(id_file, header=6, sep='\t', usecols=[0, 16, 17, 18, 19])
    return ID_col_arranger(ID)  # Return re-arranged columns


def load_time_intervals(periods_file: str) -> pd.DataFrame:
    """
    Load recording periods.
    periods_file (str): full directory name
    """
    periods = pd.read_csv(periods_file)
    return periods


def load_features(features_file: str) -> pd.DataFrame:
    """
    features_file: .csv file
    """
    features = pd.read_csv(features_file, index_col='time')
    return features


def merge_joints(IK: pd.DataFrame, ID: pd.DataFrame) -> pd.DataFrame:
    """Join kinematics and kinetics data into a single pandas dataframe along with record periods dataframe which used to filter the dataset by removing all periods where ID solution was not available (no GRF data)
    """
    # Merge kinematics and kinetics data
    joints_data = pd.merge(IK, ID, on='time', how='inner')
    # Assert no data loss
    # assert len(joints_data) == len(ID) == len(IK)
    # Reset time to zero to match EMG
    joints_data = reset_time(joints_data)
    return joints_data

def select_walking_trials(joints_data, periods):
    left = periods[['left_start', 'left_end']].dropna()
    right = periods[['right_start', 'right_end']].dropna()
    right = right.reset_index(drop=True)
    # work on left side
    left_joints = ['knee_angle_l_moment','ankle_angle_l_moment']
    right_joints = ['knee_angle_r_moment','ankle_angle_r_moment']
    joints_data.loc[:left.iloc[0,0], left_joints] = np.nan
    joints_data.loc[:right.iloc[0,0], right_joints] = np.nan
    for i in range(1, len(left)):
        previous_end = left.loc[i-1,'left_end']
        current_start = left.loc[i, 'left_start']
        joints_data.loc[previous_end:current_start, left_joints] = np.nan
    for i in range(1, len(right)):
        previous_end = right.loc[i-1,'right_end']
        current_start = right.loc[i, 'right_start']
        joints_data.loc[previous_end:current_start, right_joints] = np.nan
    return joints_data
    


def reset_time(data: pd.DataFrame) -> pd.DataFrame:
    start_time = data['time'].min()
    data['time'] = data['time'] -start_time
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


def ID_col_arranger(ID: pd.DataFrame) -> pd.DataFrame:
    """
    Arrange ID data columns to be:
        ['time', 'knee_angle_r_moment', 'knee_angle_l_moment',
                'ankle_angle_r_moment', 'ankle_angle_l_moment']
    """
    col = ID.columns
    return ID[[col[0], col[1], col[3], col[2], col[4]]]


def get_dataset(subject=None) -> None:
    # If subject number wasn't provided ask the user to manually input it
    if subject == None:
        subject = input("Please write subject number in a format XX: ")
    # Load the subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        date = subject_details[f"S{subject}"]["date"]
    # Experiment trials names
    files = ['train_01', 'train_02', 'val', 'test']
    ########################### Get I/O directories ###########################
    ik_path = f"../OpenSim/S{subject}/{date}/IK/"
    IK_files = list(map(lambda x: f"{ik_path}S{subject}_{x}_IK.mot", files))

    id_path = f"../OpenSim/S{subject}/{date}/ID/"
    ID_files = list(map(lambda x: f"{id_path}{x}/inverse_dynamics.sto", files))

    record_periods_path = f"../Outputs/S{subject}/{date}/record_periods/"
    periods_files = list(
        map(lambda x: f"{record_periods_path}S{subject}_{x}_record_periods.csv", files))

    features_path = f"../Outputs/S{subject}/{date}/EMG/"
    Features_files = list(
        map(lambda x: f"{features_path}{x}_features.csv", files))

    output_folder = f"../Dataset/S{subject}/"
    output_files = list(
        map(lambda x: f"{output_folder}{x}_dataset.csv", files))
    ########################### Loop in each trial ###########################
    for ik_file, id_file, periods_file, features_file, output_name\
            in zip(IK_files, ID_files, periods_files, Features_files, output_files):
        # Load IK data
        IK = load_IK(ik_file)
        # Load ID data
        ID = load_ID(id_file)
        # Load record interval data
        periods = load_time_intervals(periods_file)
        # Load EMG features
        features = load_features(features_file)
        # Merge IK, ID & record intervals together to create joint's dataset
        joints_data = merge_joints(IK, ID)
        # Merge EMG features with joints data and down sample joints data to match the EMG features
        joints_data = select_walking_trials(joints_data, periods)
        Dataset = merge_IO(features, joints_data)
        # Rename column and save the dataset
        new_col = {'knee_angle_r': "Right knee angle",
                   'ankle_angle_r': 'Right ankle angle',
                   'knee_angle_l': "Left knee angle",
                   'ankle_angle_l': 'Left ankle angle',
                   'ankle_angle_r_moment': "Right ankle moment",
                   'knee_angle_r_moment': "Right knee moment",
                   'ankle_angle_l_moment': "Left ankle moment",
                   'knee_angle_l_moment': "Left knee moment"}
        Dataset.rename(columns=new_col, inplace=True)
        Dataset.to_csv(output_name)


if __name__ == "__main__":
    s = "01"
    get_dataset(s)
