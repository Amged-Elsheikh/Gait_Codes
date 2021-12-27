import pandas as pd
import numpy as np
import json


def load_IK(ik_file):
    """
    Load knee and ankle joints angles
    ik_file: full extension name
    """
    IK = pd.read_csv(ik_file, header=8, sep='\t', usecols=[0, 10, 11, 17, 18])
    return IK


def load_ID(id_file):
    """
    load knee and ankle joints moments
    id_file: full extension name
    """
    ID = pd.read_csv(id_file, header=6, sep='\t', usecols=[0, 16, 17, 18, 19])
    return ID_col_arranger(ID)  # Return re-arranged columns


def load_time_intervals(periods_file):
    """
    Load recording periods.
    periods_file: full name extension
    """
    periods = pd.read_csv(periods_file, index_col="time")
    return periods


def merge_joints(IK, ID, periods):
    # Merge kinematics and kinetics data
    joints_data = pd.merge(IK, ID, on='time', how='inner')
    # Assert no data loss
    assert len(joints_data) == len(ID) == len(IK)
    # Merge the columns that tells when to make measurements (record periods)
    joints_data_with_events = pd.merge(
        joints_data, periods, on='time', how='inner')
    # Assert no data lost
    # assert len(joints_data_with_events) == len(joints_data)
    # Reset time to zero to match EMG
    joints_data_with_events = reset_time(joints_data_with_events)
    return joints_data_with_events


def load_features(features_file):
    """
    features_file: .csv file
    """
    features = pd.read_csv(features_file, index_col='time')
    return features


def merge_IO(features, joints_data):
    """
    downsampling is hold while merging by removing points
    """
    # Merge all features and joints. Down sampling is done while merging
    Dataset = pd.merge(left=features, right=joints_data,
                       on='time', how='inner')
    Dataset.set_index("time", inplace=True)
    return Dataset


def ID_col_arranger(ID):
    """
    Arrange ID data columns to be:
        ['time', 'knee_angle_r_moment', 'knee_angle_l_moment',
                'ankle_angle_r_moment', 'ankle_angle_l_moment']
    """
    col = ID.columns
    return ID[[col[0], col[1], col[3], col[2], col[4]]]


def reset_time(data):
    start_time = data['time'].min()
    data['time'] = data['time'].apply(lambda x: x-start_time)
    data['time'] = np.around(data['time'], 3)
    return data


new_labels = {'ankle_angle_l', 'knee_angle_r_moment', 'ankle_angle_r_moment',
              'knee_angle_l_moment', 'ankle_angle_l_moment', 'Unnamed: 0', 'left_side', 'right_side'}
# %%


def get_dataset(subject=None):
    if subject == None:
        subject = input("Please write subject number in a format XX: ")

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    date = subject_details[f"S{subject}"]["date"]
    # Get trials names
    files = ['train_01', 'train_02', 'val', 'test']

    ik_path = f"../OpenSim/S{subject}/{date}/IK/"
    IK_files = list(map(lambda x: f"{ik_path}S{subject}_{x}_IK.mot", files))

    id_path = f"../OpenSim/S{subject}/{date}/ID/"
    ID_files = list(map(lambda x: f"{id_path}{x}/inverse_dynamics.sto", files))

    record_periods_path = f"../Outputs/S{subject}/{date}/record_periods/"
    periods_files = list(
        map(lambda x: f"{record_periods_path}{x}_record_periods.csv", files))

    features_path = f"../Outputs/S{subject}/{date}/EMG/"
    Features_files = list(
        map(lambda x: f"{features_path}{x}_features.csv", files))

    output_folder = f"../Dataset/S{subject}/"
    output_files = list(
        map(lambda x: f"{output_folder}{x}_dataset.csv", files))

    for ik_file, id_file, periods_file, features_file, output_name\
            in zip(IK_files, ID_files, periods_files, Features_files, output_files):

        IK = load_IK(ik_file)
        ID = load_ID(id_file)
        periods = load_time_intervals(periods_file)
        features = load_features(features_file)
        joints_data = merge_joints(IK, ID, periods)
        Dataset = merge_IO(features, joints_data)
        Dataset.loc[Dataset['left_side'] == False, [
            'knee_angle_l_moment', 'ankle_angle_l_moment']] = np.nan

        Dataset.loc[Dataset['right_side'] == False, [
            'knee_angle_r_moment', 'ankle_angle_r_moment']] = np.nan

        Dataset.drop(columns=['left_side', 'right_side'],
                     inplace=True)  # Drop periods columns

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
    for s in ["01", "02", "04"]:
        get_dataset(s)
