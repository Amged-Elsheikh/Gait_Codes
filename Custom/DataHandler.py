import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler


class DataHandler:
    with open("subject_details.json", "r") as f:
        all_subjects = json.load(f)

    def __init__(self, subject, features, add_knee=True,
                 out_labels=["knee moment", "ankle moment"]):
        # Initiate the subject
        self.subject = subject
        self.features = features
        self.add_knee = add_knee
        self.out_labels = out_labels
        # get datasets directories
        trials = ["train_01", "train_02", "val", "test"]
        trials_directory = list(
            map(lambda x: f"../Dataset/S{subject}/{x}_dataset.csv", trials))
        # #Load data and store it in a dictionary
        self.data = dict()
        for trial, trial_directory in zip(trials, trials_directory):
            self.data[trial] = pd.read_csv(
                trial_directory, index_col="time")
        # Features number info

        # Select columns and normalize
        self._get_datasets_columns()
        self.model_columns = self.emg_features.copy()
        if self.add_knee:
            self.model_columns.append("knee angle")
        
        self.model_columns.extend(self.out_labels)

        for trial in self.data.keys():
            # Select columns
            self.data[trial] = self.data[trial][self.dataset_columns]
            # Scale
            self.data[trial] = self.scale(self.data[trial])
            # Take only selected columns for model
            self.data[trial] = self.data[trial][self.model_columns]

        if self.add_knee:
            self.features_num += 1

    @property
    def details(self):
        return self.all_subjects[f"S{self.subject}"]

    @property
    def weight(self):
        return self.details["weight"]

    def _get_datasets_columns(self):
        self.joints_columns = list(
            filter(lambda x: "sensor" not in x, self.data["val"]))
        self.emg_features = []
        # O(N*n) method
        for col in self.data["val"]:
            for feature in self.features:
                if feature in col:
                    self.emg_features.append(col)
        self.features_num = len(self.emg_features)
        self.dataset_columns = self.emg_features.copy()
        self.dataset_columns.extend(self.joints_columns)

    def scale(self, data):
        features_slice = slice(0, self.features_num)
        joints_data_num = len(self.joints_columns)
        angle_slice = slice(-joints_data_num, -joints_data_num//2)
        moment_slice = slice(-joints_data_num//2, None)
        # Scale features
        features_scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaler.fit(data.dropna().iloc[:100, features_slice])
        data.iloc[:, features_slice] = features_scaler.transform(
            data.iloc[:, features_slice])
        # scale angles
        angle_scaler = MinMaxScaler(feature_range=(0, 1))
        angle_scaler.fit(data.dropna().iloc[:100, angle_slice])
        data.iloc[:, angle_slice] = angle_scaler.transform(
            data.iloc[:, angle_slice])
        # Scale weight
        data.iloc[:, moment_slice] = data.iloc[:, moment_slice] / self.weight
        return data

if __name__ == "__main__":
    DataHandler("06", ["RMS","ZC","AR"])


