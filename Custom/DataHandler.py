import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from typing import *


class DataHandler:
    '''
    A custom class that will make working with data easier.
    '''
    def __init__(self, subject: str, features: List[str], sensors: List[str], add_knee=False,
                 out_labels: List[str] = ["ankle moment"], emg_type='sEMG'):

        # Initiate the subject
        if emg_type not in ['sEMG', 'DEMG']:
            raise "emg_type must be either sEMG or DEMG"
        self.subject = subject
        self.features = features
        self.sensors = sensors
        self.add_knee = add_knee
        self.out_labels = out_labels
        # get datasets directories
        trials = ["train_01", "train_02", "val", "test"]
        trials_directory = list(
            map(lambda x: f"../Dataset/{emg_type}/S{subject}/{x}_dataset.csv", trials))
        # #Load data and store it in a dictionary. The self method with the dictionary will allow accessing the dataset easily.
        self.data: Dict[str, pd.DataFrame] = dict()
        for trial, trial_directory in zip(trials, trials_directory):
            self.data[trial] = pd.read_csv(trial_directory, index_col="time")

        # Create a list of features columns
        emg_features = []
        # All features in the dataset
        features_columns = list(
            filter(lambda x: "sensor" in x, self.data["val"]))
        # Get the desired EMG features
        for sensor in self.sensors:
            for feature in self.features:
                emg_features.append(f'{sensor} {feature}')
        # Count the number of features
        self.features_num = len(emg_features)
        # initiate the columns for the models (I&O)
        self.model_columns = emg_features.copy()
        # Add the knee angle if required. Do n't increment the features number here
        if self.add_knee:
            self.model_columns.append("knee angle")
        # Add the output columns for the model columns
        self.model_columns.extend(self.out_labels)
        # Scale the data
        self._joints_columns = list(
            filter(lambda x: "sensor" not in x, self.data["val"]))
        # dataset_columns includes selected features and sensors combinations along with all joints data
        dataset_columns = emg_features.copy()
        dataset_columns.extend(self._joints_columns)
        self._is_scaler_available = False
        for trial in trials:
            # Select columns. (all joints data are inncluded)
            self.data[trial] = self.data[trial].loc[:, dataset_columns]
            # Scale
            self.data[trial] = self._scale(self.data[trial])
            # Take only selected columns for model
            self.data[trial] = self.data[trial].loc[:, self.model_columns]
        # If the knee is added, then increment the number of features by 1
        if self.add_knee:
            self.features_num += 1

    @ property
    def _subject_weight(self) -> float:
        with open("subject_details.json", "r") as f:
            return json.load(f)[f"S{self.subject}"]["weight"]

    def _scale(self, data):
        '''
        Scale the Dataset
        '''
        # Create slice objects to select different type of data
        features_slice = slice(0, self.features_num)
        joints_data_num = len(self._joints_columns)
        angle_slice = slice(-joints_data_num, -joints_data_num//2)
        moment_slice = slice(-joints_data_num//2, None)
        
        # Scale features between 0 and 1
        if not self._is_scaler_available:
            self._features_scaler = MinMaxScaler(feature_range=(0, 1))
            # The scaler will fit only data from the recording periods.
            self._features_scaler.fit(data.dropna().iloc[:300, features_slice])
        data.iloc[:, features_slice] = self._features_scaler.transform(
            data.iloc[:, features_slice])

        # scale angles
        if not self._is_scaler_available:
            self._angle_scaler = MinMaxScaler(feature_range=(0, 1))
            self._angle_scaler.fit(data.dropna().iloc[:300, angle_slice])
        data.iloc[:, angle_slice] = self._angle_scaler.transform(
            data.iloc[:, angle_slice])
        # Scale moments by subjext's weight
        data.iloc[:, moment_slice] = data.iloc[:, moment_slice] / self._subject_weight
        
        # Set the scaler value to True to avoid creating new scalers
        self._is_scaler_available = True
        return data
