from Tringo.TrignoListenerSample import TrignoListener
from Tringo import TrignoClient
from Custom.models_functions import *
from EMG_Process import *
from functools import partial
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import time
import argparse
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# from matplotlib.animation import FuncAnimation


class Device:
    # Set Device parameters
    channel_range = (1, 3)
    FEATURES_PER_SENSOR = 2
    FEATURES_NUM = FEATURES_PER_SENSOR*channel_range[1]
    TIME_STEP = 15
    sample_rate = 1111.1111
    samples_per_read = int(SLIDING_WINDOW_STRIDE * sample_rate)

    apply_np_func = partial(np.apply_along_axis, axis=0)

    def __init__(self, model_func):
        # Initialize the device
        self.dev = TrignoClient.TrignoEMG(channel_range=self.channel_range,
                                          samples_per_read=self.samples_per_read,
                                          units="mV", host="localhost")

        self.listener = TrignoListener()
        self.dev.frameListener = self.listener.on_emg

        # Initialize the model
        self.model_func = model_func
        self.model = self.model_func(ModelParameters(self.TIME_STEP,
                                                     features_num=self.FEATURES_NUM,
                                                     out_nums=1, label_width=1))

        self.model.load_weights(
            f"../Results/GM/LSTM model/S04/S04_LSTM model.hdf5")
    
    def run(self):
        self.dev.run()
        print('Waiting for Data..')
        while self.listener.data is None:
            pass

    @staticmethod
    def apply_filter(data):
        """Apply bandpass filter and notch filter to the EMG data"""
        filtered_emg = apply_notch_filter(bandpass_filter(data))
        return filtered_emg

    @staticmethod
    def remove_mean(data, mean_array):
        """Remove mean value from all Sensors. The mean array will be calculated only once durning the normalization process"""
        data = data - mean_array
        return data

    @staticmethod
    def features_functions(filtered_data, features=[get_RMS, zero_crossing]):
        """Extract EMG features from filtered EMG"""
        return np.array((feature_func(filtered_data) for feature_func in features))

    def get_features_array(self, filtered_emg, features=None):
        if features == None:
            features = np.empty(
                shape=(0, self.FEATURES_PER_SENSOR*filtered_emg.shape[1]))
        # Sliding window parameters
        start = 0
        end = self.dev.samples_per_read
        step = int(SLIDING_WINDOW_STRIDE*self.sample_rate)

        while end < len(filtered_emg):
            window_data = filtered_emg[start:end]

            window_data = self.apply_np_func(
                features_functions, arr=window_data)

            window_data = np.reshape(window_data, (1, -1), order="F")

            features = np.vstack((features, window_data))

            start += step
            end += step

        return features

    def get_normalization_parameters(self, measuring_time=10.0):
        """measuring_time in seconds"""

        # Collect and filter EMG data
        filtered_emg = self.apply_np_func(
            self.apply_filter, arr=self.listener.data.transpose())

        while filtered_emg.shape[0] < measuring_time * 1111.11:
            filtered_emg = np.vstack((filtered_emg, self.apply_np_func(
                self.apply_filter, arr=self.listener.data.transpose())))

        # remove the mean from the filtered EMG data
        self.mean_array = np.mean(filtered_emg, 0, keepdims=True)
        filtered_emg = self.remove_mean(filtered_emg, mean_array)
        # Get features array from the filtered EMG data
        features = self.get_features_array(self.dev, filtered_emg)

        self.scaler = MinMaxScaler()
        self.scaler.fit(features)

    def EMG_2_features(self, emg_deque: deque):
        # emg = np.array(np.vstack(i) for i in emg_deque)
        emg = np.empty((0, self.channel_range[1]))
        for i in emg_deque:
            emg = np.vstack((emg, i))

        filtered_emg = self.remove_mean(self.apply_np_func(self.apply_filter,
                                                 arr=emg), mean_array)

        features = self.get_features_array(self.dev, filtered_emg)
        features = scaler.transform(features)
        return features

    def Animate_ankle_moment(ankle_moment_queue):
        plt.cla()
        plt.plot(ankle_moment_queue)


class ModelParameters:
    def __init__(self, input_width, features_num, out_nums, label_width):
        self.input_width = input_width
        self.features_num = features_num
        self.out_nums = out_nums
        self.label_width = label_width


if __name__ == "__main__":
    # Collect Normalization data
    ankle_plotter = Device()
    ankle_plotter.run()
    print("collecting Normalize data will start after 5 seconds")
    time.sleep(0.5)
    print("start collecting normalization data")
    # Get normalization data
    scaler, mean_array = get_normalization_parameters(dev, measuring_time=2)

    # Initialize EMG Queue
    number_of_mini_windows = WINDOW_LENGTH / \
        SLIDING_WINDOW_STRIDE + (TIME_STEP-1)
    emg_deque = deque(maxlen=int(number_of_mini_windows))

    # Fill EMG deque
    print("Filling the queue collection started")
    while len(emg_deque) < number_of_mini_windows:
        emg_deque.append(listener.data.transpose())
    # Start looping after filling the Queue
    Quit = False
    ankle_deque = []
    # plt.figure()
    # plt.show()
    while not Quit:
        try:
            # The queue is already filled
            features = EMG_2_features(emg_deque, mean_array, scaler)
            # Reshape the features to fit the model
            features = np.reshape(features, (1, -1, FEATURES_NUM))
            # Apply DL model to the features
            ankle_moment = model.predict(features)[0][0][0]
            # print(ankle_moment)
            # ankle_deque.append(ankle_moment)
            # Update emg queue
            emg_deque.append(listener.data.transpose())
        except:
            dev.stop()
            Quit = True
