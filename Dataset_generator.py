# In[1]:
print("Hello World")
# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from functools import partial


# In[7]:
def joints_filter(data):
    f = 6 # Filter frequency
    fs = 100 # Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=2, Wn=low_pass, btype='lowpass')
    columns = data.columns[1:]
    for col in columns:
        data[col] = filtfilt(b2, a2, data[col])
    return data

def load_IK(ik_file):
    """
    ik_file: full extension name
    """
    IK = pd.read_csv(ik_file ,header=8, sep='\t', usecols=[0,10,11,17,18])
    return IK

def load_ID(id_file):
    """
    id_file: file_name/.sto
    """
    ID = pd.read_csv(id_file, header=6, sep='\t', usecols=[0,16,17,18,19])
    ID = ID_col_arranger(ID)
    return ID

def load_time_intervals(periods_file):
    periods = pd.read_csv(periods_file, index_col="time")
    return periods

def merge_joints(IK,ID,periods):
    # Merge kinematics and kinetics data
    joints_data = pd.merge(IK, ID, on='time', how='inner')
    # Assert no data loss
    assert len(joints_data)==len(ID)==len(IK)
    # low pass filtering OpenSim data
    joints_data = joints_filter(joints_data)
    # Merge the columns that tells when to make measurements
    joints_data_with_events = pd.merge(joints_data, periods, on='time', how='inner')
    # Assert no data lost
    assert len(joints_data_with_events)==len(joints_data)
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
    Dataset = pd.merge(left=features, right=joints_data, on='time', how='inner')
    assert len(Dataset)==len(features)
    Dataset.set_index("time", inplace=True)
    return Dataset

def ID_col_arranger(ID):
    col = ID.columns
    return ID[[col[0], col[1], col[3], col[2], col[4]]]

def reset_time(data):
    start_time = data['time'].min()
    data['time'] = data['time'].apply(lambda x:x-start_time)
    data['time'] = np.around(data['time'], 2)
    return data


# In[8]:


def get_dataset(subject=None):
    if subject==None:
        subject = input("Please write subject number in a format XX: ")
    
    files = [f'S{subject}_test', f'S{subject}_train_01', f'S{subject}_train_02', f'S{subject}_val']
    settings = pd.read_csv(f"../settings/dataset_settings/S{subject}_dataset_settings.csv", header=None)
    
    ik_path = settings.iloc[0,1]
    IK_files = list(map(lambda x: f"{ik_path}{x}_IK.mot", files))

    id_path = settings.iloc[1,1]
    ID_files = list(map(lambda x: f"{id_path}{x}/inverse_dynamics.sto", files))
    
    record_periods_path = settings.iloc[2,1]
    periods_files = list(map(lambda x: f"{record_periods_path}{x}_record_periods.csv", files))
    
    features_path = settings.iloc[3,1]
    Features_files = list(map(lambda x: f"{features_path}{x}_features.csv", files))

    output_folder = settings.iloc[4,1]
    output_files = list(map(lambda x: f"{output_folder}{x}_dataset.csv", files))
    
    for ik_file, id_file, periods_file, features_file, output_name\
        in zip(IK_files,ID_files, periods_files, Features_files, output_files):
        
        IK = load_IK(ik_file)
        ID = load_ID(id_file)
        periods = load_time_intervals(periods_file)
        features = load_features(features_file)
        joints_data = merge_joints(IK,ID, periods)
        Dataset = merge_IO(features, joints_data)
        Dataset.loc[Dataset['left_side']==False,
                    ['knee_angle_l_moment','ankle_angle_l_moment']]=np.nan
        
        Dataset.loc[Dataset['right_side']==False,
                    ['knee_angle_r_moment','ankle_angle_r_moment']]=np.nan
        
        Dataset.drop(columns=['left_side', 'right_side'], inplace=True)
        Dataset.to_csv(output_name)

# In[9]:


get_dataset("02")

