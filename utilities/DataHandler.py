import json
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataHandler:
    """
    A custom class that will make working with data easier.
    """

    def __init__(
        self,
        subject: str,
        features: List[str],
        sensors: List[str],
        label="ankle moment",
        emg_type="sEMG",
    ):
        # Initiate the subject
        if emg_type not in ["sEMG", "DEMG"]:
            raise "emg_type must be either sEMG or DEMG"
        self.subject = subject
        self.features = features
        self.label = label
        # Create a list of features columns
        self.columns = [
            f"{sensor} {feature}" for sensor in sensors for feature in features
        ]
        self.features_num = len(self.columns)
        self.columns.append(label)
        # get datasets directories
        trials = ["train_01", "train_02", "val", "test"]
        trials_directory = [
            Path("..", "Dataset", emg_type, f"S{subject}", f"{x}_dataset.csv")
            for x in trials
        ]
        self.data = {
            trial: pd.read_csv(
                trial_directory, index_col="time", usecols=["time"] + self.columns
            )
            for trial, trial_directory in zip(trials, trials_directory)
        }
        for trial in trials:
            self.data[trial] = self._scale(self.data[trial])

    @property
    def _weight(self) -> float:
        with open("subject_details.json", "r") as f:
            return json.load(f)[f"S{self.subject}"]["weight"]

    def _scale(self, data):
        # Scale moments by subject's weight
        data.loc[:, self.label] = data.loc[:, self.label] / self._weight
        # Scale features between 0 and 1
        if not hasattr(self, '_has_scaler'):
            self._features_scaler = MinMaxScaler(feature_range=(0, 1))
            self._features_scaler.fit(data.dropna().iloc[:300, : self.features_num])
            self._has_scaler = True
        data.iloc[:, : self.features_num] = self._features_scaler.transform(
            data.iloc[:, : self.features_num]
        )
        return data


if __name__ == "__main__":
    a = DataHandler("08", ["WL"], [f"sensor {i}" for i in [6, 7, 8, 9]])
    a
