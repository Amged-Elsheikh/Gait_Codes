import numpy as np
import tensorflow as tf
from tensorflow import keras
from Custom.DataHandler import DataHandler
tf.random.set_seed(42)


class WindowGenerator:
    def __init__(
            self, dataHandler,
            input_width=10, label_width=None,
            shift=1, batch_size=128,):

        self.batch_size = batch_size
        self.dataHandler = dataHandler
        self.out_labels = self.dataHandler.out_labels
        self.out_nums = len(self.out_labels)
        ####### Store the raw data #######
        self.train_01_df = self.dataHandler.trials_data['train_01']
        self.train_02_df = self.dataHandler.trials_data['train_02']
        self.val_df = self.dataHandler.trials_data['val']
        self.test_df = self.dataHandler.trials_data['test']

        ####### Window parameters #########
        # For more details about this part in code go to https://www.tensorflow.org/tutorials/structured_data/time_series
        # Set the the number of timesteps I/O
        self.input_width = input_width
        self.label_width = label_width
        if self.label_width == None:
            self.label_width = input_width
        # Shift parameters shows the difference in timesteps between last input and last output
        self.shift = shift
        # Create one window that hold all inputs and outputs timesteps
        self.total_window_size = input_width + shift
        # Time level window
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    @ property
    def features_columns(self):
        return self.dataHandler.model_columns[:-self.output_num]

    @ property
    def features_num(self):
        return self.dataHandler.features_num

    def IO_window(self, data):
        """
        Create tf time-series data set [Batch_size, timestep, features/labels].
        Can handle each side-data separately or provide a full dataset if you want to work with a unidirectional model
        """
        return keras.preprocessing.timeseries_dataset_from_array(
            data = data, targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1, shuffle = False,
            batch_size = self.batch_size,
        )

    def split_window(self, features):
        # Take all EMG features and knee angle column
        # Shape is [Batch_size, timestep, features/labels]
        inputs=features[:, self.input_slice, :-self.out_nums]
        # Predict ankle angle & torque
        labels=features[:, self.labels_slice, -self.out_nums:]
        # Slicing doesn't preserve static shape information, so set the shapes manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def preprocessing(
        self, ds, remove_nan = True, shuffle = False,  batch_size = None, drop_reminder = False
    ):
        """Will process batched dataset according to the inputs, then returned cached and prefetched the tf.data

        Args:
            ds ([tf.data]): [Shape is [Batch_size, timestep, features/labels]]
            remove_nan (bool, optional): [description]. Defaults to True.
            batch_size ([int], optional): [batch size, if None, default window object batch size will be used]. Defaults to None.
            shuffle (bool, optional): [Shuffle the data]. Defaults to False.
            drop_reminder (bool, optional): [drop the last batch if data are not enough]. Defaults to False.

        Returns:
            [prefetched tf.data]:
        """
        ds=ds.unbatch()
        if remove_nan:
            def filter_nan(_, y):
                return not tf.reduce_any(tf.math.is_nan(y))

            ds=ds.filter(filter_nan)
        ds=ds.cache()
        if shuffle:
            ds=ds.shuffle(buffer_size = 16000, reshuffle_each_iteration=True)
        if not batch_size:
            batch_size=self.batch_size
        ds=ds.batch(batch_size, drop_remainder = drop_reminder)
        ds=ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    @ property
    def train_dataset(self):
        """Make the training dataset for the indiviual models from train_01 and train_02.

        Returns:
            prefetched tf.data: training set
        """
        # Prepare each trail sides
        trail_01=self.IO_window(self.train_01_df)
        trail_02=self.IO_window(self.train_02_df)
        # stack trails datasets
        train_ds=trail_01.concatenate(trail_02)
        # Split window data to input and output and store results
        train_ds=train_ds.map(self.split_window)
        # Shufffle the train dataset
        train_ds=self.preprocessing(
            train_ds, shuffle = True, drop_reminder = True, remove_nan = True
        )
        return train_ds

    @property
    def val_dataset(self):
        """Make the validation dataset for the indiviual models.

        Returns:
            prefetched tf.data: validation set
        """
        # prepare sides
        val_ds = self.IO_window(self.val_df)
        # Split window data to input and output and store results
        val_ds = val_ds.map(self.split_window)
        # Make the batch size as big as possible
        val_ds = self.preprocessing(
            val_ds,
            batch_size=16000,
            shuffle=False,
            drop_reminder=False,
            remove_nan=True,
        )
        return val_ds

    @property
    def evaluation_set(self):
        """Make the evaluation dataset for the indiviual models. In validation set, remove the NaN values but in evaluation set NaN values are important.

        Returns:
            prefetched tf.data: evaluation set
        """
        # prepare sides
        test_ds = self.IO_window(self.test_df)
        # Split window data to input and output and store results
        test_ds = test_ds.map(self.split_window)
        # Make the batch size as big as possible
        test_ds = self.preprocessing(
            test_ds,
            batch_size=16000,
            shuffle=False,
            drop_reminder=False,
            remove_nan=False,
        )
        return test_ds

    def get_gm_train_val_dataset(self):
        """Get the training and validation datasets for GMs. test dataset is used for the training. self.preprocessing will be implemented in the GM python file

        Returns:
            [type]: [description]
        """
        trail_01 = self.IO_window(self.train_01_df)
        trail_02 = self.IO_window(self.train_02_df)
        val_ds = self.IO_window(self.val_df)
        test_ds = self.IO_window(self.test_df)
        # stack trails datasets
        # Add validation set to the training set and use test set for validation
        train_ds = trail_01.concatenate(trail_02)
        train_ds = train_ds.concatenate(test_ds)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        val_ds = val_ds.map(self.split_window)
        return train_ds, val_ds

    def make_dataset(self):
        """Make train/val/test datasets for individual models. Test dataset can be used to test GM also.
        """
        return self.train_dataset, self.val_dataset, self.evaluation_set

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"input timestep: {self.input_width}",
                f"output timestep: {self.label_width}",
            ]
        )
