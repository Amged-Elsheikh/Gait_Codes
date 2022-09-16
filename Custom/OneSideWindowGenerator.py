import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from Custom.DataHandler import DataHandler
from typing import *
tf.random.set_seed(42)


class WindowGenerator:
    def __init__(self, dataHandler: DataHandler,
                 input_width=10, label_width=1,
                 shift=1, batch_size=128, is_general_model=False):
        '''
        Window object will make it easier to work with the time series dataset, allowing more flexability
        when adjusting the parameters of the input and output sizes and lengths.
        For more details about this part in code go to https://www.tensorflow.org/tutorials/structured_data/time_series
        '''
        self.dataHandler = dataHandler
        self.batch_size = batch_size
        self.out_labels = dataHandler.out_labels
        self.out_nums = len(self.out_labels)
        self._is_general_model = is_general_model
        ####### Store the raw data #######
        self.train_01_df = self.dataHandler.data['train_01']
        self.train_02_df = self.dataHandler.data['train_02']
        self.val_df = self.dataHandler.data['val']
        self.test_df = self.dataHandler.data['test']
        ####### Window parameters #########
        # Set the the timesteps
        self.input_width = input_width
        self.label_width = label_width
        # Shift parameters shows the difference in timesteps between last input and last output
        self.shift = shift
        # Create one window that hold all inputs and outputs timesteps
        self.total_window_size = input_width + shift
        # features slicer (slice through the time)
        self.input_timestep_slice = slice(0, input_width)
        # Find the output's startig timestep
        self.label_start_timestep = self.total_window_size - self.label_width
        # Output slicer (slice through the time)
        self.labels_timestep_slice = slice(self.label_start_timestep, None)

    @ property
    def features_columns(self) -> list[str]:
        return self.dataHandler.model_columns[:self.features_num]

    @ property
    def features_num(self) -> int:
        return self.dataHandler.features_num

    def IO_window(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Create tf time-series data set [Batch_size, timestep, features/labels].
        Can handle each side-data separately or provide a full dataset if you want to work with a unidirectional model
        """
        return keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1, shuffle=False,
            batch_size=self.batch_size,)

    def split_window(self, features: tf.data.Dataset) -> List[tf.data.Dataset]:
        '''Take all EMG features and knee angle column. Shape is [Batch_size, timestep, features/labels]'''
        inputs = features[:, self.input_timestep_slice, :-self.out_nums]
        # Predict ankle angle & torque
        labels = features[:, self.labels_timestep_slice, -self.out_nums:]
        # Slicing doesn't preserve static shape information, so set the shapes manually
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    @classmethod
    def preprocessing(cls, ds: tf.data.Dataset, remove_nan=True,
                      shuffle=False, batch_size=None, drop_reminder=False) -> tf.data.Dataset:
        """Will process batched dataset according to the inputs, then returned cached and prefetched the tf.data
        """
        # Data are batched, unbatch it
        ds = ds.unbatch()
        # Remove the nan values for training (otherwise model won't train)
        if remove_nan:
            def filter_nan(_, y):
                return not tf.reduce_any(tf.math.is_nan(y))

            ds = ds.filter(filter_nan)
        # The dataset size is small, so cache in the memory to speed the code runtime
        ds = ds.cache()
        # Shuffle the data
        if shuffle:
            ds = ds.shuffle(buffer_size=16000, reshuffle_each_iteration=True)
        # create batches
        ds = ds.batch(batch_size, drop_remainder=drop_reminder)
        # Prefetch to speed the code runtime
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    @ property
    def train_dataset(self) -> tf.data.Dataset:
        """Make the training dataset for the indiviual models from train_01 and train_02.
        """
        # Prepare each trail sides
        trail_01 = self.IO_window(self.train_01_df)
        trail_02 = self.IO_window(self.train_02_df)
        # stack trails datasets
        train_ds = trail_01.concatenate(trail_02)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        if not self._self._is_general_model:
            # Process the training set (shuffle, and remove nan)
            train_ds = self.preprocessing(train_ds, shuffle=True, batch_size=self.batch_size,
                                      drop_reminder=True, remove_nan=True)
        return train_ds

    @property
    def val_dataset(self) -> tf.data.Dataset:
        """Make the validation dataset for the indiviual models."""
        # Create timeseries dataset
        val_ds = self.IO_window(self.val_df)
        # Split window data to inputs and outputs
        val_ds = val_ds.map(self.split_window)
        if not self._self._is_general_model:
            # Make the batch size as big as possible to fit the whole dataset in a single batch
            val_ds = self.preprocessing(val_ds, shuffle=False, batch_size=160000,
                                    drop_reminder=False, remove_nan=True)
        return val_ds

    @property
    def evaluation_set(self) -> tf.data.Dataset:
        """Make the evaluation dataset for the indiviual models. In validation set, remove the NaN values but in evaluation set NaN values can stay.
        """
        # Create timeseries dataset
        test_ds = self.IO_window(self.test_df)
        # Split window data to inputs and outputs
        test_ds = test_ds.map(self.split_window)
        # Make the batch size as big as possible to fit the whole dataset in a single batch
        test_ds = self.preprocessing(test_ds, shuffle=False, batch_size=160000,
                                     drop_reminder=False, remove_nan=False)
        return test_ds

    def make_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Make train/val/test datasets for individual models. Test dataset can be used to test GM also.
        """
        return self.train_dataset, self.val_dataset, self.evaluation_set
