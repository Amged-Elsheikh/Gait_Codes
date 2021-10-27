#!/usr/bin/env python
# coding: utf-8
"""
1. Filter data
2. remove mean
3. differentiate
4. remove artifacts
5. get features
"""
# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from statsmodels.tsa.ar_model import AutoReg
# from sklearn.preprocessing import StandardScaler as std
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
plt.rcParams["figure.figsize"] = [14, 10]

# In[26]:


def get_emg_files(subject, outputs_path):
    # Get inputs names
    inputs_names = ["test", "train_01", "train_02", "val"]
    # add path, subject number and file extension
    inputs_names = list(map(lambda x: f"S{subject}_{x}_EMG.csv", inputs_names))
    # Get outputs names
    output_files = list(map(lambda x: f"{outputs_path}{x}", inputs_names))
    output_files = list(
        map(lambda x: x.replace("_EMG", "_features"), output_files))
    return inputs_names, output_files


def load_emg_data(inputs_path, emg_file):
    emg = pd.read_csv(f"{inputs_path}{emg_file}", header=12)
    emg.columns = emg.columns.str.replace(": EMG.*", "", regex=True)
    emg.columns = emg.columns.str.replace("Trigno IM ", "", regex=True)
    emg.columns = emg.columns.str.replace("X [s]", "time", regex=False)
    emg.set_index("time", inplace=True)
    # work with EMG signals from the Shank only
    emg = emg[[f"sensor {i}" for i in range(7, 13)]]
    return emg


def apply_notch_filter(data):
    fs = 1/0.0009  # Hz
    f0 = 50  # Notched frequancy
    Q = 30  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, data)


def apply_filter(emg, order=4, lowband=20, highband=450):
    fs = 1/0.0009  # Hz
    low_pass = lowband/(fs*0.5)
    hig_pass = highband/(fs*0.5)
    b, a = signal.butter(N=order, Wn=[low_pass, hig_pass], btype="bandpass")
    return signal.filtfilt(b, a, emg)


def remove_outlier(data, detect_factor=30, remove_factor=15):
    detector = data.apply(np.abs).mean()*detect_factor
    data = data.apply(lambda x: x if np.abs(x) < detector else x/remove_factor)
    return data


def process_emg_signal(emg, remove_artifacts=True):
    # filter the signals
    filtered_emg = emg.apply(apply_filter)
    filtered_emg = filtered_emg.apply(apply_notch_filter)
    # Remove The mean
    filtered_emg = filtered_emg-filtered_emg.mean()
    # Remove artifacts
    if remove_artifacts:
        for col in filtered_emg.columns:
            filtered_emg.loc[:, col] = remove_outlier(filtered_emg.loc[:, col])
    # differentiate the signal
    DEMG = filtered_emg.apply(np.gradient)/0.0009
    return DEMG


def plot_all_emg(emg, file_name=None):
    m = 2  # number of columns
    n = int(len(emg.columns)/2)  # number of raws
    plt.figure(file_name)
    for i in range(n):
        plt.subplot(n, m, i+1)
        plt.plot(emg.index, emg.iloc[:, 2*i], emg.index, emg.iloc[:, 2*i+1])
    plt.suptitle(file_name)
    plt.draw()

# ### Features Functions

# In[27]:


def get_AR_coeffs(data, num_coeff=6):
    model = AutoReg(data, lags=num_coeff)  # old_names=True)
    model_fit = model.fit()
    return model_fit.params


def get_MAV(data):
    return sum(abs(n) for n in data)/len(data)


def zero_crossing(data):
    # returns the indexes of where ZC appear
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    return len(zero_crossings)


def get_RMS(data):
    return np.sqrt(sum(n*n for n in data)/len(data))


def wave_length(data):
    return np.sum(abs(np.diff(data)))

# In[]:


def get_features(DEMG):
    step = 0.05
    dataset = pd.DataFrame()
    time_limit = max(DEMG.index)
    print(f"time_limit: {time_limit}s")
    for EMG_num in range(len(DEMG.columns)):
        start = 0
        end = 0.250
        coeff = []
        MAV = []
        RMS = []
        # ZC = []
        EMG_label = f"sensor {EMG_num+7}"
        sensor_data = DEMG[EMG_label]
        # Extract features
        while (end < time_limit):
            window_data = sensor_data[(DEMG.index >= start) & (
                DEMG.index < end)].to_numpy()
            # Get the AR coefficients
            # coeff.append(get_AR_coeffs(window_data))
            # Get the MAV
            MAV.append(get_MAV(window_data))
            # Get RMS
            RMS.append(get_RMS(window_data))
            # Get Zero-Crossing
            # ZC.append(zero_crossing(window_data))
            # Update window
            start = start + step
            end = end + step

        # coeff = np.array(coeff)
        # ZC = np.array(ZC)
        RMS = np.array(RMS)
        MAV = np.array(MAV)
        dataset_temp = pd.DataFrame(
            {f'DEMG{EMG_num+1}_MAV': MAV, f'DEMG{EMG_num+1}_RMS': RMS})
        dataset = pd.concat([dataset, dataset_temp], axis=1)
#         print(f"{EMG_label} done")

    dataset['time'] = [np.around(step*i, 3) for i in range(len(dataset))]
    dataset.set_index("time", inplace=True)
    # dataset.describe()
    return dataset


def plot_MAV(dataset, emg_file):
    MAV_columns = [f'DEMG{i+1}_MAV' for i in range(6)]
    MAV_data = dataset[MAV_columns]
    plot_all_emg(MAV_data, emg_file)

# In[48]:


def emg_to_features(subject, remove_artifacts=True):
    if not subject:
        subject = input("Please input subject number in XX format: ")
    # Load settings
    settings = pd.read_csv(
        f"../settings/emg_settings/S{subject}_emg_settings.csv", header=None)
    inputs_path = settings.iloc[0, 1]
    outputs_path = settings.iloc[1, 1]
    # Get EMG files directories
    inputs_names, output_files = get_emg_files(subject, outputs_path)
    for emg_file, output_file in zip(inputs_names, output_files):
        # Preprocessing
        emg = load_emg_data(inputs_path, emg_file)  # Load data
        DEMG = process_emg_signal(emg, remove_artifacts)  # preprocess the data
        dataset = get_features(DEMG)  # get features
        # save dataset
        dataset.to_csv(output_file)
        # Plot data
        plot_MAV(dataset, emg_file)


emg_to_features(subject="02", remove_artifacts=True)
plt.show()
