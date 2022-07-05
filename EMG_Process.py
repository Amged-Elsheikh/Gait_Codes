"""
1. Filter data
2. remove mean
3. differentiate
4. remove artifacts
5. get features
"""
import json
import re
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg

# Setups
simplefilter(action='ignore', category=FutureWarning)
WINDOW_LENGTH = 0.2
SLIDING_WINDOW_STRIDE = 0.05


def get_emg_files(subject: str, trials: list) -> str:
    """This function will return I/O directories stored in two lists.
    """
    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)[f"S{subject}"]
        date = subject_details["date"]

    # Get I/O data directory
    inputs_path = f"../Data/S{subject}/{date}/EMG/"
    outputs_path = f"../Outputs/S{subject}/{date}/EMG/"
    # add path, subject number and file extension
    inputs_names = list(
        map(lambda x: f"{inputs_path}S{subject}_{x}_EMG.csv", trials))
    # Get outputs names
    output_files = list(
        map(lambda x: f"{outputs_path}{x}_features.csv", trials))
    return inputs_names, output_files


def load_emg_data(subject: str, emg_file: str) -> pd.DataFrame:
    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)[f"S{subject}"]["emg_sync"]

    emg = pd.read_csv(emg_file, header=0)
    # Rename time column
    emg.columns = emg.columns.str.replace("X[s]", "time", regex=False)
    # Set time column as the index
    emg.set_index("time", inplace=True)
    # Keep EMG only
    emg = emg.filter(regex="EMG")
    # Rename the column
    emg.columns = emg.columns.str.replace(": EMG.*", "", regex=True)
    emg.columns = emg.columns.str.replace("Trigno IM ", "", regex=True)
    # Subset EMG data
    trial = re.sub(".*S[0-9]*_", "", emg_file)
    trial = re.sub("_[a-zA-Z].*", "", trial)
    start = subject_details[trial]["start"]
    end = start + subject_details[trial]["length"]
    emg = emg.loc[(start <= emg.index) & (emg.index <= end)]
    emg = emg - emg.mean()
    return emg


def remove_outlier(data, detect_factor=10, remove_factor=20):
    for col in data.columns:
        column_data = data.loc[:, col]
        detector = column_data.apply(np.abs).mean()*detect_factor
        data.loc[:, col] = column_data.apply(
            lambda x: x if np.abs(x) < detector else x/remove_factor)
    return data


def segmant(emg, start, end):
    return emg.loc[(emg.index >= start) & (emg.index <= end), :]


def bandpass_filter(window, order=4, lowband=20, highband=450):
    fs = 1/0.0009  # Hz
    low_pass = lowband/(fs*0.5)
    hig_pass = highband/(fs*0.5)
    b, a = signal.butter(N=order, Wn=[low_pass, hig_pass], btype="bandpass")
    return signal.filtfilt(b, a, window, axis=0)


def notch_filter(window):
    fs = 1/0.0009  # sampling frequancy in Hz
    f0 = 50  # Notched frequancy
    Q = 30  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, window, axis=0)


def emg_filter(window):
    return notch_filter(bandpass_filter(window))


def get_ZC(window):
    # returns the indexes of where ZC appear return a tuple with length of 1
    ZC = [len(np.where(np.diff(np.signbit(window[:, col])))[0])
          for col in range(np.shape(window)[-1])]
    return np.array(ZC)


def get_RMS(window):
    return np.sqrt(sum(n*n for n in window)/len(window))


def get_MAV(window):
    return sum(abs(n) for n in window)/len(window)


def wave_length(window):
    return np.sum(abs(np.diff(window, axis=0)), axis=0)


def get_AR_coeffs(window, num_coeff=6):
    first = True
    for col in range(np.shape(window)[-1]):
        model = AutoReg(window[:, col], lags=num_coeff, old_names=True)
        model_fit = model.fit()
        if first:
            parameters = model_fit.params[1:]
            first = False
        else:
            parameters = np.vstack((parameters, model_fit.params[1:]))
    return parameters


def get_single_window_features(filtered_window, features_names, ar_order):

    features_dict = {"RMS": get_RMS,
                     "MAV": get_MAV,
                     "WL": wave_length,
                     "ZC": get_ZC, }

    features_function = [features_dict[feature]
                         for feature in features_names if "AR" not in feature]

    features = np.vstack([foo(filtered_window)
                         for foo in features_function]).transpose()

    if ar_order:
        features = np.hstack(
            (features, get_AR_coeffs(filtered_window, ar_order)))

    return features.flatten()


def process_emg(emg, features_names=["RMS", "MAV", "WL", "ZC"], ar_order=4):
    """
    EMG has zero mean and no artifacts. This function will segmant the data, filter it and apply features extraction methods to return the dataset.
    """
    if ar_order:
        features_names.extend([f"AR{i}" for i in range(1, ar_order+1)])

    df_col = []
    for emg_num in range(1, emg.shape[1]+1):
        df_col.extend([f"sensor {emg_num} {f}" for f in features_names])

    dataset = pd.DataFrame(columns=df_col)

    start = 0
    end = WINDOW_LENGTH
    time_limit = max(emg.index)
    print(f"time_limit: {time_limit}s")
    while end <= time_limit:
        # Segmant the data
        window = segmant(emg, start, end)
        # Filter segmanted data
        window = emg_filter(window)
        #
        window = np.diff(window, axis=0)/0.0009
        # Get features
        features = get_single_window_features(window, features_names, ar_order)
        # Update data frame
        dataset.loc[len(dataset)] = features
        start += SLIDING_WINDOW_STRIDE
        end += SLIDING_WINDOW_STRIDE

    dataset['time'] = [np.around(SLIDING_WINDOW_STRIDE*i + WINDOW_LENGTH, 3)
                       for i in range(len(dataset))]

    dataset.set_index("time", inplace=True)
    return dataset


def emg_to_features(subject=None, trials=["train_01", "train_02", "val", "test"], features_names=["RMS", "MAV", "WL", "ZC"], ar_order=4):
    if not subject:
        subject = input("Please input subject number: ")
        subject = f"{int(subject):02d}"
    # Get EMG files directories
    inputs_names, output_files = get_emg_files(subject, trials)

    for emg_file, output_file, trial in zip(inputs_names, output_files, trials):
        # Load data
        emg = load_emg_data(subject, emg_file)
        # reset time
        emg.index -= np.round(min(emg.index), 5)
        # Remove artifacts
        emg = remove_outlier(emg)  # Un-efficient method used
        # preprocess the data to get features
        dataset = process_emg(emg, features_names, ar_order)
        # save dataset
        dataset.to_csv(output_file)

        # Plot data
        # plot_feature(dataset, "RMS", trial)
if __name__ == "__main__":
    DEMG = True
    trials = ["train_01", "train_02", "val", "test"]
    features_names = ["RMS", "MAV", "WL", "ZC"]
    ar_order = 4

    for subject in [6, 8, 9, 10, 13, 14]:
        # subject = input("Please input subject number: ")
        subject = f"{int(subject):02d}"
        emg_to_features(subject, trials, features_names, ar_order)

        try:
            # If all subject data files exisit, the dataset will be automatically generated/updated
            from Dataset_generator import *
            get_dataset(subject)
            print("Dataset file been updated successfully.")
        except:
            pass

    # plt.show()
