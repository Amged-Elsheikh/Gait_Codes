import numpy as np
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(self, train_01_df, train_02_df, val_df, test_df,
                 input_width=10, label_width=None, shift=1,
                 batch_size=128, add_knee=False, out_labels=["ankle moment"]):

        self.batch_size = batch_size
        self.out_labels = out_labels
        self.out_nums = len(self.out_labels)
        self.add_knee = add_knee
        ####### Store the raw data #######

        self.train_01_df = train_01_df
        self.train_02_df = train_02_df
        self.val_df = val_df
        self.test_df = test_df
        ####### Seperate left left and right side columns #######
        # Get number of features per sensor
        self.features_per_sensor = 0
        for col in test_df.columns:
            if "DEMG1" in col:
                self.features_per_sensor += 1
            else:
                break

        inputs_col = train_01_df.columns[:-8].values
        sensors_num = len(inputs_col)//self.features_per_sensor
        left_side_col = []
        right_side_col = []
        for i in range(sensors_num//2):
            left_side_col.extend(
                inputs_col[self.features_per_sensor*2*i: self.features_per_sensor*2*i+self.features_per_sensor].tolist())

            right_side_col.extend(
                inputs_col[self.features_per_sensor*2*i+self.features_per_sensor: self.features_per_sensor*2*i+self.features_per_sensor*2].tolist())
        # append knee angle as input and outputs
        if self.add_knee:
            left_side_col.append("Left knee angle")
            right_side_col.append("Right knee angle")
        self.features_num = len(left_side_col)  # used for input layer

        left_side_col.extend(list(map(lambda x: f"Left {x}", self.out_labels)))
        right_side_col.extend(
            list(map(lambda x: f"Right {x}", self.out_labels)))
        self.left_side_col = left_side_col
        self.right_side_col = right_side_col

        ####### Window parameters #########
        self.input_width = input_width
        if label_width == None:
            self.label_width = input_width
        else:
            self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        # Time level window
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

        self.left_val_data = self.left_side_data(val_df)
        self.right_val_data = self.right_side_data(val_df)
        self.left_test_data = self.left_side_data(test_df)
        self.right_test_data = self.right_side_data(test_df)

    def left_side_data(self, data):
        """
        data should be one of the dataset stored in the class object
        """
        return data[self.left_side_col]

    def right_side_data(self, data):
        """
        data should be one of the dataset stored in the class object
        """
        return data[self.right_side_col]

    def split_window(self, features):
        # Take all EMG features and knee angle column
        # [Batch_size, timestep, features/labels]
        inputs = features[:, self.input_slice, :-self.out_nums]
        # Predict ankle angle & torque
        labels = features[:, self.labels_slice, -self.out_nums:]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def IO_window(self, data):
        return keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None,
                                                                 sequence_length=self.total_window_size,
                                                                 sequence_stride=1, shuffle=False,
                                                                 batch_size=self.batch_size)

    def prepare_sides(self, data):
        """
        Seperate each side data and create windows
        data: Pandas df

        return: concatenated tf.data
        """
        # Seperate sides
        left = self.left_side_data(data).values
        right = self.right_side_data(data).values
        # Create window datasets
        left = self.IO_window(left)
        right = self.IO_window(right)
        # Stack and return datasets
        return left.concatenate(right)

    def get_train_dataset(self):
        # Prepare each trail sides
        trail_01 = self.prepare_sides(self.train_01_df)
        trail_02 = self.prepare_sides(self.train_02_df)

        # stack trails datasets
        train_ds = trail_01.concatenate(trail_02)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        # Shufffle the train dataset
        train_ds = self.preprocessing(
            train_ds, shuffle=True, drop_reminder=True)
        return train_ds

    def get_val_dataset(self):
        # prepare sides
        val_ds = self.prepare_sides(self.val_df)
        # Split window data to input and output and store results
        val_ds = val_ds.map(self.split_window)
        # Make the batch size as big as possible
        val_ds = self.preprocessing(
            val_ds, batch_size=16000, shuffle=False, remove_nan=True)
        return val_ds

    def get_evaluation_set(self):
        # prepare sides
        test_ds = self.prepare_sides(self.test_df)
        # Split window data to input and output and store results
        test_ds = test_ds.map(self.split_window)
        # Make the batch size as big as possible
        test_ds = self.preprocessing(
            test_ds, batch_size=16000, shuffle=False, remove_nan=False)
        return test_ds

    def get_gm_train_val_dataset(self):
        trail_01 = self.prepare_sides(self.train_01_df)
        trail_02 = self.prepare_sides(self.train_02_df)
        val_ds = self.prepare_sides(self.val_df)
        test_ds = self.prepare_sides(self.test_df)
        # stack trails datasets
        # Add validation set to the training set and use test set for validation
        train_ds = trail_01.concatenate(trail_02)
        train_ds = train_ds.concatenate(test_ds)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        val_ds = val_ds.map(self.split_window)
        return train_ds, val_ds

    def preprocessing(self, ds, batch_size=None, shuffle=False, remove_nan=True, drop_reminder=False):
        ds = ds.unbatch()
        if remove_nan:
            def filter_nan(_, y): return not tf.reduce_any(tf.math.is_nan(y))
            ds = ds.filter(filter_nan)
            pass
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=16000, reshuffle_each_iteration=True)
        if not batch_size:
            batch_size = self.batch_size
        ds = ds.batch(batch_size, drop_remainder=drop_reminder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def make_dataset(self):
        train_set = self.get_train_dataset()
        val_set = self.get_val_dataset()
        test_set = self.get_evaluation_set()
        return train_set, val_set, test_set

    def __repr__(self):
        return '\n'.join([
            f"Total window size: {self.total_window_size}",
            f"input timestep: {self.input_width}",
            f"output timestep: {self.label_width}"])
