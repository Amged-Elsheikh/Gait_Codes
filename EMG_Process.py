"""
1. Filter data
2. remove mean
3. differentiate
4. remove artifacts
5. get features
"""
import json
from statsmodels.tsa.ar_model import AutoReg
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from warnings import simplefilter

# Setups
simplefilter(action='ignore', category=FutureWarning)

WINDOW_LENGTH = 0.25
SLIDING_WINDOW_STRIDE = 0.05


def get_emg_files(subject: str, inputs_path: str, outputs_path: str, trials: list) -> str:
    """This function will return I/O directories stored in two lists.
    """
    # add path, subject number and file extension
    inputs_names = list(
        map(lambda x: f"{inputs_path}S{subject}_{x}_EMG.csv", trials))
    # Get outputs names
    output_files = list(
        map(lambda x: f"{outputs_path}{x}_features.csv", trials))
    return inputs_names, output_files


def load_emg_data(subject: str, emg_file: str, trial: str) -> pd.DataFrame:
    delsys = pd.read_csv(emg_file, header=0)
    # Rename time column
    delsys.columns = delsys.columns.str.replace("X[s]", "time", regex=False)
    # Set time column as the index
    delsys.set_index("time", inplace=True)
    # Keep EMG only
    emg = delsys.filter(regex="EMG")
    # Rename the column
    emg.columns = emg.columns.str.replace(": EMG.*", "", regex=True)
    emg.columns = emg.columns.str.replace("Trigno IM ", "", regex=True)
    # Subset EMG data
    start = subject_details[f"S{subject}"]["emg_sync"][trial]["start"]
    end = start + subject_details[f"S{subject}"]["emg_sync"][trial]["length"]
    return emg.loc[(start <= emg.index) & (emg.index <= end)]


def process_emg_signal(emg: pd.DataFrame, remove_artifacts=True) -> pd.DataFrame:
    # filter the signals
    filtered_emg = emg.apply(bandpass_filter)
    filtered_emg = filtered_emg.apply(apply_notch_filter)
    # Ensure signal has zero mean
    filtered_emg = filtered_emg-filtered_emg.mean()
    # Remove artifacts
    if remove_artifacts:
        for col in filtered_emg.columns:
            filtered_emg.loc[:, col] = remove_outlier(filtered_emg.loc[:, col])
    return filtered_emg


def bandpass_filter(emg, order=4, lowband=20, highband=450):
    fs = 1/0.0009  # Hz
    low_pass = lowband/(fs*0.5)
    hig_pass = highband/(fs*0.5)
    b, a = signal.butter(N=order, Wn=[low_pass, hig_pass], btype="bandpass")
    return signal.filtfilt(b, a, emg)


def apply_notch_filter(data):
    fs = 1/0.0009  # sampling frequancy in Hz
    f0 = 50  # Notched frequancy
    Q = 30  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, data)


def remove_outlier(data, detect_factor=20, remove_factor=15):
    detector = data.apply(np.abs).mean()*detect_factor
    data = data.apply(lambda x: x if np.abs(x) < detector else x/remove_factor)
    return data


# Features Functions
def get_AR_coeffs(data, num_coeff=6):
    model = AutoReg(data, lags=num_coeff)  # old_names=True)
    model_fit = model.fit()
    return model_fit.params


def get_MAV(data):
    return sum(abs(n) for n in data)/len(data)


def zero_crossing(data):
    # returns the indexes of where ZC appear return a tuple with length of 1
    zero_crossings = np.where(np.diff(np.signbit(data)))
    return len(zero_crossings[0])


def get_RMS(data):
    return np.sqrt(sum(n*n for n in data)/len(data))


def wave_length(data):
    return np.sum(abs(np.diff(data)))


def features_functions(DEMG: pd.DataFrame):
    global WINDOW_LENGTH, SLIDING_WINDOW_STRIDE
    dataset = pd.DataFrame()
    time_limit = max(DEMG.index)
    print(f"time_limit: {time_limit}s")
    for EMG_num in range(1, len(DEMG.columns)+1):
        start = 0
        end = WINDOW_LENGTH
        coeff = []
        MAV = []
        RMS = []
        ZC = []
        EMG_label = f"sensor {EMG_num}"
        sensor_data = DEMG[EMG_label]
        # Extract features
        while (end < time_limit):
            window_data = sensor_data[(DEMG.index >= start) & (
                DEMG.index < end)].to_numpy()
            # Get the AR coefficients
            coeff.append(get_AR_coeffs(window_data))
            # Get the MAV
            MAV.append(get_MAV(window_data))
            # Get RMS
            RMS.append(get_RMS(window_data))
            # #Get Zero-Crossing
            ZC.append(zero_crossing(window_data))
            # Update window
            start += SLIDING_WINDOW_STRIDE
            end += SLIDING_WINDOW_STRIDE

        coeff = np.array(coeff)
        ZC = np.array(ZC)
        RMS = np.array(RMS)
        # MAV = np.array(MAV)

        dataset_temp = pd.DataFrame({f'DEMG{EMG_num}_ZC': ZC,
                                     f'DEMG{EMG_num}_RMS': RMS,
                                     f'DEMG{EMG_num}_MAV': MAV,
                                     f'DEMG{EMG_num}_AR1': coeff[:, 1],
                                     f'DEMG{EMG_num}_AR2': coeff[:, 2],
                                     f'DEMG{EMG_num}_AR3': coeff[:, 3],
                                     f'DEMG{EMG_num}_AR4': coeff[:, 4],
                                     f'DEMG{EMG_num}_AR5': coeff[:, 5],
                                     f'DEMG{EMG_num}_AR6': coeff[:, 6]})

        dataset = pd.concat([dataset, dataset_temp], axis=1)
#         print(f"{EMG_label} done")

    dataset['time'] = [np.around(SLIDING_WINDOW_STRIDE*i, 3) for i in range(len(dataset))]
    dataset.set_index("time", inplace=True)
    # dataset.describe()
    return dataset


def plot_all_emg(emg, plot_time_range=None, file_name=None):
    plt.figure(file_name)
    muscles = ["Tibialis Anterior", "Gastrocnemius Medialis", "Soleus"]
    # number of columns
    m = 1 
    # number of raws
    n = int(len(muscles))
    if plot_time_range == None:
        plot_time_range = (emg.index[0], emg.index[-1])
    start = plot_time_range[0]
    end = plot_time_range[1]

    for i in range(n):
        plt.subplot(n, m, i+1)
        plt.plot(emg.index, emg.iloc[:, 2*i], emg.index, emg.iloc[:, 2*i+1])
        plt.title(muscles[i])
        plt.xlim((start, end+0.01))
        plt.ylabel("Magnitude")
        if i == 0:
            plt.legend(["Left", "Right"], loc="upper right")
    plt.xlabel("Time [s]")
    plt.suptitle(file_name)
    plt.tight_layout()
    plt.draw()


def plot_RMS(dataset, emg_file):
    RMS_columns = [f'DEMG{i+1}_RMS' for i in range(sensors_num)]
    RMS_data = dataset[RMS_columns]
    plot_all_emg(RMS_data, None, emg_file)


def emg_to_features(subject=None, remove_artifacts=True):
    if not subject:
        subject = input("Please input subject number in XX format: ")

    date = subject_details[f"S{subject}"]["date"]
    # Get I/O data directory
    inputs_path = f"../Data/S{subject}/{date}/EMG/"
    outputs_path = f"../Outputs/S{subject}/{date}/EMG/"
    # Set experements trials
    trials = ["test", "train_01", "train_02", "val"]
    # Get EMG files directories
    inputs_names, output_files = get_emg_files(
        subject, inputs_path, outputs_path, trials)
    for emg_file, output_file, trial in zip(inputs_names, output_files, trials):
        # Load data
        emg = load_emg_data(subject, emg_file, trial)
        # reset time
        emg.index -= np.round(min(emg.index), 4)
        # preprocess the data
        filtered_emg = process_emg_signal(emg, remove_artifacts)
        # Get features
        dataset = features_functions(filtered_emg)
        # save dataset
        dataset.to_csv(output_file)
        # Plot data
        plot_RMS(dataset, emg_file)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    plt.rcParams["figure.figsize"] = [14, 10]

    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)

    sensors_num = 6  # Very important for loading the data

    for s in ["01", "02", "04"]:
        emg_to_features(s, remove_artifacts=True)
        try:
            # If all subject data files exisit, the dataset will be automatically generated
            from Dataset_generator import *
            get_dataset(s)
            print("Dataset file been updated successfully.")
        except:
            pass
        plt.close()
