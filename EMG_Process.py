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

# %%


def get_emg_files(subject: str, date: str, trials: list) -> str:
    """This function will return I/O directories stored in two lists.
    """
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
    emg = emg.loc[(start <= emg.index) & (emg.index <= end)]
    emg = emg - emg.mean()
    return emg


# %%
def remove_outlier(data, detect_factor=20, remove_factor=15):
    for col in data.columns:
        column_data = data.loc[:,col]
        detector = column_data.apply(np.abs).mean()*detect_factor
        data.loc[:,col] = column_data.apply(lambda x: x if np.abs(x) < detector else x/remove_factor)
    return data


def segmant(emg, start, end):
    return emg.loc[(emg.index >= start) & (emg.index <= end), :]

# %%


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

# %%
# Features Functions


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
    return np.sum(abs(np.diff(window)))


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


# %%

def get_single_window_features(filtered_window):
    features_function = [get_ZC, get_RMS, get_MAV]
    features = np.vstack([foo(filtered_window) for foo in features_function]).transpose()
    return np.hstack((features, get_AR_coeffs(filtered_window))).flatten()


def process_emg(emg):
    """
    EMG has zero mean and no artifacts. This function will segmant the data, filter it and apply features extraction methods to return the dataset.
    """
    features_names = ["ZC", "RMS", "MAV"]
    features_names.extend([f"AR{i}" for i in range(1, 7)])
    df_col = []
    for emg_num in range(1, emg.shape[1]+1):
        df_col.extend([f"sensor {emg_num} {f}" for f in features_names])
    dataset = pd.DataFrame(columns=df_col)

    # Remove artifacts
    # emg = remove_outlier(emg)
    start = 0
    end = WINDOW_LENGTH
    time_limit = max(emg.index)
    print(f"time_limit: {time_limit}s")
    while end <= time_limit:
        # Segmant the data
        window = segmant(emg, start, end)
        # Filter segmanted data
        filtered_window = emg_filter(window)
        # Get features
        features = get_single_window_features(filtered_window)
        # Update data frame
        dataset.loc[len(dataset)] = features
        start += SLIDING_WINDOW_STRIDE
        end += SLIDING_WINDOW_STRIDE
        
    dataset['time'] = [np.around(SLIDING_WINDOW_STRIDE*i + WINDOW_LENGTH, 3)
                       for i in range(len(dataset))]

    dataset.set_index("time", inplace=True)
    return dataset
# %%

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
    RMS_data = dataset.filter(regex="RMS")
    plot_all_emg(RMS_data, None, emg_file)


def emg_to_features(subject=None):
    if not subject:
        subject = input("Please input subject number in XX format: ")

    date = subject_details[f"S{subject}"]["date"]
    # Set experements trials
    trials = ["test", "train_01", "train_02", "val"]
    # Get EMG files directories
    inputs_names, output_files = get_emg_files(subject, date, trials)

    for emg_file, output_file, trial in zip(inputs_names, output_files, trials):
        # Load data
        emg = load_emg_data(subject, emg_file, trial)
        # reset time
        emg.index -= np.round(min(emg.index), 5)
        # Remove artifacts
        emg = remove_outlier(emg) # Un-efficient method used
        # preprocess the data to get features
        dataset = process_emg(emg)
        # save dataset
        dataset.to_csv(output_file)
        # Plot data
        # plot_RMS(dataset, emg_file)


# %%
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    plt.rcParams["figure.figsize"] = [14, 10]

    with open("subject_details.json", "r") as file:
        subject_details = json.load(file)

    for s in ["03",]:
        emg_to_features(s)
        try:
            # If all subject data files exisit, the dataset will be automatically generated/updated
            from Dataset_generator import *
            get_dataset(s)
            print("Dataset file been updated successfully.")
        except:
            pass

    plt.show()