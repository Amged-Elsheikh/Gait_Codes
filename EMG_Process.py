import json
import re
from typing import List, Dict, Union, Callable
from warnings import simplefilter
from pathlib import Path
from functools import partial
from collections import OrderedDict
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
from Dataset_generator import get_dataset

# Setups
simplefilter(action="ignore", category=FutureWarning)
# Set constant to be used by other modules.
WINDOW_LENGTH = 0.2  # The length of the sliding window in seconds
SLIDING_WINDOW_STRIDE = 0.05  # Sliding window stride in seconds


def get_emg_files(subject: str, trials: List[str], useDEMG: bool):
    """This function will return I/O directories stored in two lists."""
    # Load the experiment's setups
    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)[subject]
    # Get the date of the experiment
    date = subject_details["date"]
    # Get I/O data folders
    inputs_path = Path("..", "Data", subject, date, "EMG")
    if useDEMG:
        outputs_path = Path("..", "Outputs", subject, date, "DEMG")
    else:
        outputs_path = Path("..", "Outputs", subject, date, "sEMG")
    inputs_dir = [inputs_path / f"{subject}_{x}_EMG.csv" for x in trials]
    output_dir = [outputs_path / f"{x}_features.csv" for x in trials]
    return inputs_dir, output_dir


def load_emg_data(subject: str, emg_file: Path) -> pd.DataFrame:
    # Load experiment setups
    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)[subject]
    emg_sync_data: Dict[Dict[str, float]] = subject_details["emg_sync"]
    # Load delsys data
    emg = pd.read_csv(emg_file)
    emg = emg.rename(columns={"X[s]": "time"})
    emg = emg.filter(regex=r"EMG|time")
    emg.columns = emg.columns.str.replace(": EMG.*", "", regex=True)
    emg.columns = emg.columns.str.replace("Trigno IM ", "", regex=True)
    trial = re.sub(".*S[0-9]*_", "", str(emg_file))
    trial = re.sub("_[a-zA-Z].*", "", trial)
    # Trim the data to keep the experement period only
    start = emg_sync_data[trial]["start"]
    end = start + emg_sync_data[trial]["length"]
    emg = emg.loc[emg["time"].between(start, end)]
    emg = emg.set_index("time")
    # Ensure that the data will have a zero mean
    emg = emg - emg.mean()
    # reset time
    emg.index -= np.round(min(emg.index), 5)
    return emg


def remove_outlier(
    data: pd.DataFrame, detect_factor: float = 10, remove_factor: float = 20
):
    # loop in each column
    detector = data.abs().mean() * detect_factor
    data.loc[:] = np.where(np.abs(data) < detector, data, data / remove_factor)
    return data


def emg_filter(window: pd.DataFrame):
    return notch_filter(bandpass_filter(window))


def bandpass_filter(
    window: Union[pd.DataFrame, np.ndarray],
    order: int = 4,
    lowband: int = 20,
    highband: int = 450,
):
    fs = 1 / 0.0009  # Hz
    low_pass = lowband / (fs * 0.5)
    hig_pass = highband / (fs * 0.5)
    b, a = signal.butter(N=order, Wn=[low_pass, hig_pass], btype="bandpass")
    return signal.filtfilt(b, a, window, axis=0)


def notch_filter(window: Union[pd.DataFrame, np.ndarray]):
    fs = 1 / 0.0009  # sampling frequancy in Hz
    f0 = 50  # Notched frequancy
    Q = 30  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, window, axis=0)


def get_single_window_features(window: np.ndarray, features_dict: Dict[str, Callable]):
    features = [features_dict[feature](window)
                for feature in features_dict.keys()]
    if "AR" in features_dict:
        f1 = np.vstack(features[:-1]).transpose()
        features = np.hstack((f1, features[-1]))
    else:
        features = np.vstack(features).transpose()
    return features.flatten()


def get_ZC(window: np.ndarray) -> np.ndarray:
    # returns the indexes of where ZC appear return a tuple with length of 1
    ZC = [
        len(np.where(np.diff(np.signbit(window[:, col])))[0])
        for col in range(np.shape(window)[-1])
    ]
    return np.array(ZC)


def get_RMS(window: np.ndarray) -> np.ndarray:
    return np.sqrt(sum(n * n for n in window) / len(window))


def get_MAV(window: np.ndarray) -> np.ndarray:
    return sum(abs(n) for n in window) / len(window)


def wave_length(window: np.ndarray) -> np.ndarray:
    return np.sum(abs(np.diff(window, axis=0)), axis=0)


def get_AR_coeffs(window: np.ndarray, num_coeff=4):
    num_cols = window.shape[-1]
    parameters = np.empty((0, num_coeff))
    for col in range(num_cols):
        model = AutoReg(window[:, col], num_coeff)
        model_fit = model.fit()
        parameters = np.vstack((parameters, model_fit.params[1:]))
    return parameters


def process_emg(
    emg: pd.DataFrame,
    features_dict: Dict[str, Callable],
    features: List[str] = ["RMS", "MAV", "WL", "ZC"],
    useDEMG=True,
):
    """
    This function will segmant the data, filter it and apply features
    extraction methods to return the dataset.
    """
    num_windows = (max(emg.index) - WINDOW_LENGTH) // SLIDING_WINDOW_STRIDE + 1
    indices = np.arange(num_windows) * SLIDING_WINDOW_STRIDE
    windows = np.array(
        [
            emg.loc[start:end].to_numpy()
            for start, end in zip(indices, indices + WINDOW_LENGTH)
        ],
        dtype=object,
    )
    df = [window_process(window, features_dict, useDEMG) for window in windows]
    cols = [f"{sensor} {f}" for sensor in emg for f in features]
    df = pd.DataFrame(df, columns=cols)
    df.index = np.around(df.index * SLIDING_WINDOW_STRIDE + WINDOW_LENGTH, 2)
    df.index.name = "time"
    return df


def window_process(window: np.ndarray, features_dict, useDEMG=True):
    window = notch_filter(bandpass_filter(window))
    if useDEMG:
        window = np.diff(window, axis=0) / 0.0009
    features_row = get_single_window_features(window, features_dict)
    return features_row


def emg_to_features(
    subject: Union[str, int],
    trials: List[str],
    features_dict: Dict[str, Callable],
    ar_order=4,
    useDEMG=True,
):
    # Get inputs and outputs directories
    try:
        subject = f"S{int(subject):02d}"
    except Exception as e:
        raise f"Exception {e}: Subject variable should be a number"

    # Add AR to the features list
    features = list(features_dict.keys())
    if ar_order > 0:
        features.extend([f"AR{i}" for i in range(1, ar_order + 1)])
        features_dict["AR"] = partial(get_AR_coeffs, num_coeff=ar_order)
    inputs_names, output_files = get_emg_files(subject, trials, useDEMG)
    Parallel(n_jobs=-1, verbose=10)(
        delayed(parallel_process)(
            subject, emg_file, features_dict, features, useDEMG, output_file
        )
        for emg_file, output_file in zip(inputs_names, output_files)
    )
    return


def parallel_process(
    subject: str,
    emg_file: Path,
    features_dict: Dict[str, Callable],
    features: List[str],
    useDEMG: bool,
    output_file: Path,
):
    # Load data
    emg = load_emg_data(subject, emg_file)
    # Remove artifacts
    emg = remove_outlier(emg, detect_factor=10, remove_factor=20)
    # preprocess the data to get features
    dataset = process_emg(emg, features_dict, features, useDEMG)
    # save dataset
    dataset.to_csv(output_file)
    return


if __name__ == "__main__":
    # ########################## prepare setups ###########################
    trials = ["train_01", "train_02", "val", "test"]
    # Set the features names
    features_dict: Dict[str, Callable] = OrderedDict(
        [
            ("RMS", get_RMS),
            ("MAV", get_MAV),
            ("WL", wave_length),
            ("ZC", get_ZC),
        ]
    )
    # set ordewr to zero if you do not want AR feature
    ar_order = 4
    # Use DEMG or sEMG features
    useDEMG = True
    subject = input("Please input subject number: ")
    # ########################## Convert ###########################
    emg_to_features(subject, trials, features_dict, ar_order, useDEMG)
    try:
        # If all subject data files exisit, the dataset generated
        get_dataset(subject, useDEMG)
        print("Dataset files updated successfully.")
    except Exception:
        print("Dataset files not updated.")
