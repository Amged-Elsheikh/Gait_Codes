from Tringo.TrignoListenerSample import TrignoListener
from Tringo import TrignoClient
from Custom.PlottingFunctions import *
import EMG_Process
from functools import partial
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight')


class Device:
    # Set Device parameters
    channel_range = (1, 3)
    FEATURES_PER_SENSOR = 2
    FEATURES_NUM = FEATURES_PER_SENSOR*channel_range[1]
    TIME_STEP = 15
    sample_rate = 1111.1111  # Hz for the sensor
    time_trigger = EMG_Process.SLIDING_WINDOW_STRIDE
    samples_per_read = int(time_trigger * sample_rate)
    WINDOW_LENGTH = int(EMG_Process.WINDOW_LENGTH * sample_rate)

    apply_np_func = partial(np.apply_along_axis, axis=0)

    def __init__(self, features_functions=[EMG_Process.get_RMS, EMG_Process.zero_crossing]):
        self.features_functions = features_functions
        # Initialize the device
        self.dev = TrignoClient.TrignoEMG(channel_range=self.channel_range,
                                          samples_per_read=self.samples_per_read,
                                          units="mV", host="localhost")

        self.listener = TrignoListener()
        self.dev.frameListener = self.listener.on_emg

        # Initialize the model

    def initialize_DL_model(self, model_func):
        self.model_func = model_func
        self.model = self.model_func(ModelParameters(self.TIME_STEP,
                                                     features_num=self.FEATURES_NUM,
                                                     out_nums=1, label_width=1))

        self.model.load_weights(
            f"../Results/GM/LSTM model/S01/S01_LSTM model.hdf5")

    def run(self):
        self.dev.run()
        print('Waiting for Data..')
        while self.listener.data is None:
            pass

    def stop(self):
        self.dev.stop()

    def collect_new_data(self, emg_deque: deque):
        emg_deque.append(self.listener.data.transpose())

    def get_normalization_parameters(self, measuring_time=10.0):
        """measuring_time in seconds"""
        current_time = time.perf_counter()
        # Collect and filter EMG data
        filtered_emg = self.apply_np_func(
            self.filtering, arr=self.listener.data.transpose())
        # keep filling data for the remaining of the measuring_time period.
        while filtered_emg.shape[0] < measuring_time * 1111.11:
            while time.perf_counter() - current_time < self.time_trigger:
                pass
            new_data = self.apply_np_func(
                self.filtering, arr=self.listener.data.transpose())
            filtered_emg = np.vstack((filtered_emg, new_data))
            current_time = time.perf_counter()

        # remove the mean from the filtered EMG data
        self.mean_array = np.mean(filtered_emg, 0, keepdims=True)
        filtered_emg = self.remove_mean(filtered_emg)
        # Get features array from the filtered EMG data
        features = self.get_features_array(filtered_emg)

        self.scaler = MinMaxScaler()
        self.scaler.fit(features)

    def get_features_array(self, filtered_emg):
        # Sliding window parameters
        start = 0
        end = self.WINDOW_LENGTH
        step = self.samples_per_read
        # initialize features array.
        features = np.empty(
            shape=(0, self.FEATURES_PER_SENSOR*filtered_emg.shape[1]))
        while end < len(filtered_emg):
            window_data = filtered_emg[start:end]
            window_data = np.array(
                [self.apply_features(window_data[:, i]) for i in range(3)])
            window_data = window_data.flatten()
            features = np.vstack((features, window_data))
            start += step
            end += step

        return features

    def EMG_2_features(self, emg_deque: deque):
        # Unroll queue to numpy array
        emg = np.empty((0, self.channel_range[1]))
        for i in emg_deque:
            emg = np.vstack((emg, i))

        filtered_emg = self.remove_mean(
            self.apply_np_func(self.filtering, arr=emg))

        features = self.get_features_array(filtered_emg)
        features = self.scaler.transform(features)
        return features

    def Animate_ankle_moment(ankle_moment_queue):
        plt.cla()
        plt.plot(ankle_moment_queue)

    @staticmethod
    def filtering(data):
        """Apply bandpass filter and notch filter to the EMG data"""
        filtered_emg = EMG_Process.apply_notch_filter(
            EMG_Process.bandpass_filter(data))
        return filtered_emg

    def remove_mean(self, data):
        """Remove mean value from all Sensors. The mean array will be calculated only once durning the normalization process"""
        return (data - self.mean_array)

    def apply_features(self, window_data):
        """Extract EMG features from filtered EMG"""
        return np.array([feature_func(window_data) for feature_func in self.features_functions])


class ModelParameters:
    def __init__(self, input_width, features_num, out_nums, label_width):
        self.input_width = input_width
        self.features_num = features_num
        self.out_nums = out_nums
        self.label_width = label_width

if __name__ == "__main__":
    # Collect Normalization data
    tringo_emg = Device()
    tringo_emg.run()
    print("collecting Normalize data will start after 5 seconds")
    time.sleep(0.5)
    print("start collecting normalization data")
    # Get normalization data
    tringo_emg.get_normalization_parameters(measuring_time=5)
    # Select the prediction model
    tringo_emg.initialize_DL_model(model_func=create_lstm_model)
    # # Initialize EMG Queue
    base_window_boxes = EMG_Process.WINDOW_LENGTH/EMG_Process.SLIDING_WINDOW_STRIDE
    base_window_points = int(base_window_boxes*Device.samples_per_read)
    number_of_mini_windows = Device.TIME_STEP - 1  # Remove the base window
    number_of_points_in_mini_window = number_of_mini_windows * Device.samples_per_read
    emg_deque = deque(
        maxlen=int(number_of_mini_windows + base_window_boxes))

    # Fill EMG deque with mini-windows
    print("start filling the queue")
    current_time = time.perf_counter()
    while len(emg_deque) < emg_deque.maxlen:
        emg_deque.append(tringo_emg.listener.data.transpose())
        while time.perf_counter() - current_time < tringo_emg.time_trigger:
            pass
    
    # Create a csv file to hold the data
    fieldnames = ["time", "ankle_moment [N.m/Kg]"]
    with open('live_ankle_data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    t = 0
    # # Start looping after filling the Queue
    Quit = False
    # ankle_deque = []
    # # plt.figure()
    # # plt.show()
    trigger = EMG_Process.SLIDING_WINDOW_STRIDE
    while not Quit:
        try:
            current_time = time.perf_counter()
            # Update emg queue
            tringo_emg.collect_new_data(emg_deque)
            # Extract Features
            features = tringo_emg.EMG_2_features(emg_deque)
            # Reshape the features to fit the model
            features = np.reshape(features, (1, -1, Device.FEATURES_NUM))
            # Apply DL model to the features
            ankle_moment = tringo_emg.model.predict(features)[0][0][0]
            # Update the plot
            with open('live_ankle_data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                info = {
                    "time": t,
                    "ankle_moment [N.m/Kg]": ankle_moment
                }

                csv_writer.writerow(info)
                t += trigger
                print(ankle_moment)
            # Avoid overlapping by waiting for the buffer
            while time.perf_counter() - current_time < trigger:
                pass
        except:
            tringo_emg.stop()
            Quit = True
