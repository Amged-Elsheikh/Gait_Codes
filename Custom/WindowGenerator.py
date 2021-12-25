<<<<<<< HEAD
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(
        self,
        train_01_df,
        train_02_df,
        val_df,
        test_df,
        input_width=10,
        label_width=None,
        shift=1,
        batch_size=128,
        add_knee=False,
        out_labels=["ankle moment"],
    ):

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
        # Get left and right side columns names
        inputs_col = train_01_df.columns[:-8].values
        sensors_num = len(inputs_col) // self.features_per_sensor
        self.left_side_col = []
        self.right_side_col = []
        for i in range(sensors_num // 2):
            self.left_side_col.extend(
                inputs_col[
                    self.features_per_sensor * 2 * i : self.features_per_sensor * 2 * i
                    + self.features_per_sensor
                ].tolist()
            )

            self.right_side_col.extend(
                inputs_col[
                    self.features_per_sensor * 2 * i
                    + self.features_per_sensor : self.features_per_sensor * 2 * i
                    + self.features_per_sensor * 2
                ].tolist()
            )
        # append knee angle as input if add_knee is not None nor False
        if self.add_knee:
            self.left_side_col.append("Left knee angle")
            self.right_side_col.append("Right knee angle")
        # Get the total number of features to use in the input layer so that we do not need to adjust models every time.
        self.features_num = len(self.left_side_col)
        # Add each side labels
        self.left_side_col.extend(list(map(lambda x: f"Left {x}", self.out_labels)))
        self.right_side_col.extend(list(map(lambda x: f"Right {x}", self.out_labels)))

        ####### Window parameters #########
        # For more details about this part in code go to https://www.tensorflow.org/tutorials/structured_data/time_series

        self.input_width = input_width
        if label_width == None:
            self.label_width = input_width
        else:
            self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        # Time level window
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.left_val_data = self.left_side_data(val_df)
        self.right_val_data = self.right_side_data(val_df)
        self.left_test_data = self.left_side_data(test_df)
        self.right_test_data = self.right_side_data(test_df)

    def left_side_data(self, data):
        """Receive the dataset and return a dataset that contains left-side data only.
        """
        return data[self.left_side_col]

    def right_side_data(self, data):
        """Receive the dataset and return a dataset that contains right-side data only.
        """
        return data[self.right_side_col]

    def IO_window(self, data):
        """ Create tf time-series data set [Batch_size, timestep, features/labels]. Can handle each side-data separately or provide a full dataset if you want to work with a unidirectional model
        """
        return keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def prepare_sides(self, data):
        """Separate each side of data and create timesteps batches
        data: is a pd dataframe that contains left and right sides data.
        return: concatenated tf.data
        """
        # Seperate sides
        left = self.left_side_data(data).values
        right = self.right_side_data(data).values
        # Create window datasets
        left = self.IO_window(left)
        right = self.IO_window(right)
        # Stack sides and return datasets
        return left.concatenate(right)

    def split_window(self, features):
        # Take all EMG features and knee angle column
        # Shape is [Batch_size, timestep, features/labels]
        inputs = features[:, self.input_slice, : -self.out_nums]
        # Predict ankle angle & torque
        labels = features[:, self.labels_slice, -self.out_nums :]
        # Slicing doesn't preserve static shape information, so set the shapes manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def preprocessing(
        self, ds, remove_nan=True, shuffle=False,  batch_size=None, drop_reminder=False
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
        ds = ds.unbatch()
        if remove_nan:

            def filter_nan(_, y):
                return not tf.reduce_any(tf.math.is_nan(y))

            ds = ds.filter(filter_nan)
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=16000, reshuffle_each_iteration=True)
        if not batch_size:
            batch_size = self.batch_size
        ds = ds.batch(batch_size, drop_remainder=drop_reminder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def get_train_dataset(self):
        """Make the training dataset for the indiviual models from train_01 and train_02.

        Returns:
            prefetched tf.data: training set
        """
        # Prepare each trail sides
        trail_01 = self.prepare_sides(self.train_01_df)
        trail_02 = self.prepare_sides(self.train_02_df)

        # stack trails datasets
        train_ds = trail_01.concatenate(trail_02)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        # Shufffle the train dataset
        train_ds = self.preprocessing(
            train_ds, shuffle=True, drop_reminder=True, remove_nan=True
        )
        return train_ds

    def get_val_dataset(self):
        """Make the validation dataset for the indiviual models.

        Returns:
            prefetched tf.data: validation set
        """
        # prepare sides
        val_ds = self.prepare_sides(self.val_df)
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

    def get_evaluation_set(self):
        """Make the evaluation dataset for the indiviual models. In validation set, remove the NaN values but in evaluation set NaN values are important.

        Returns:
            prefetched tf.data: evaluation set
        """
        # prepare sides
        test_ds = self.prepare_sides(self.test_df)
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

    def make_dataset(self):
        """Make train/val/test datasets for individual models. Test dataset can be used to test GM also.
        """
        train_set = self.get_train_dataset()
        val_set = self.get_val_dataset()
        test_set = self.get_evaluation_set()
        return train_set, val_set, test_set

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"input timestep: {self.input_width}",
                f"output timestep: {self.label_width}",
            ]
        )

=======
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(self, train_01_df, train_02_df, val_df, test_df,
                 input_width=10, label_width=None, shift=1,
                 batch_size=128, add_knee=False, out_labels=["ankle moment"],):

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
        # Get left and right side columns names
        inputs_col = train_01_df.columns[:-8].values
        sensors_num = len(inputs_col) // self.features_per_sensor
        self.left_side_col = []
        self.right_side_col = []
        for i in range(sensors_num // 2):
            self.left_side_col.extend(
                inputs_col[
                    self.features_per_sensor * 2 * i: self.features_per_sensor * 2 * i
                    + self.features_per_sensor
                ].tolist()
            )

            self.right_side_col.extend(
                inputs_col[
                    self.features_per_sensor * 2 * i
                    + self.features_per_sensor: self.features_per_sensor * 2 * i
                    + self.features_per_sensor * 2
                ].tolist()
            )
        # append knee angle as input if add_knee is not None nor False
        if self.add_knee:
            self.left_side_col.append("Left knee angle")
            self.right_side_col.append("Right knee angle")
        # Get the total number of features to use in the input layer so that we do not need to adjust models every time.
        self.features_num = len(self.left_side_col)
        # Add each side labels
        self.left_side_col.extend(
            list(map(lambda x: f"Left {x}", self.out_labels)))
        self.right_side_col.extend(
            list(map(lambda x: f"Right {x}", self.out_labels)))

        ####### Window parameters #########
        # For more details about this part in code go to https://www.tensorflow.org/tutorials/structured_data/time_series

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
        """Receive the dataset and return a dataset that contains left-side data only.
        """
        return data[self.left_side_col]

    def right_side_data(self, data):
        """Receive the dataset and return a dataset that contains right-side data only.
        """
        return data[self.right_side_col]

    def IO_window(self, data):
        """ Create tf time-series data set [Batch_size, timestep, features/labels]. Can handle each side-data separately or provide a full dataset if you want to work with a unidirectional model
        """
        return keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def prepare_sides(self, data):
        """Separate each side of data and create timesteps batches
        data: is a pd dataframe that contains left and right sides data.
        return: concatenated tf.data
        """
        # Seperate sides
        left = self.left_side_data(data).values
        right = self.right_side_data(data).values
        # Create window datasets
        left = self.IO_window(left)
        right = self.IO_window(right)
        # Stack sides and return datasets
        return left.concatenate(right)

    def split_window(self, features):
        # Take all EMG features and knee angle column
        # Shape is [Batch_size, timestep, features/labels]
        inputs = features[:, self.input_slice, : -self.out_nums]
        # Predict ankle angle & torque
        labels = features[:, self.labels_slice, -self.out_nums:]
        # Slicing doesn't preserve static shape information, so set the shapes manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def preprocessing(
        self, ds, remove_nan=True, shuffle=False,  batch_size=None, drop_reminder=False
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
        ds = ds.unbatch()
        if remove_nan:

            def filter_nan(_, y):
                return not tf.reduce_any(tf.math.is_nan(y))

            ds = ds.filter(filter_nan)
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=16000, reshuffle_each_iteration=True)
        if not batch_size:
            batch_size = self.batch_size
        ds = ds.batch(batch_size, drop_remainder=drop_reminder)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def get_train_dataset(self):
        """Make the training dataset for the indiviual models from train_01 and train_02.

        Returns:
            prefetched tf.data: training set
        """
        # Prepare each trail sides
        trail_01 = self.prepare_sides(self.train_01_df)
        trail_02 = self.prepare_sides(self.train_02_df)

        # stack trails datasets
        train_ds = trail_01.concatenate(trail_02)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        # Shufffle the train dataset
        train_ds = self.preprocessing(
            train_ds, shuffle=True, drop_reminder=True, remove_nan=True
        )
        return train_ds

    def get_val_dataset(self):
        """Make the validation dataset for the indiviual models.

        Returns:
            prefetched tf.data: validation set
        """
        # prepare sides
        val_ds = self.prepare_sides(self.val_df)
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

    def get_evaluation_set(self):
        """Make the evaluation dataset for the indiviual models. In validation set, remove the NaN values but in evaluation set NaN values are important.

        Returns:
            prefetched tf.data: evaluation set
        """
        # prepare sides
        test_ds = self.prepare_sides(self.test_df)
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

    def make_dataset(self):
        """Make train/val/test datasets for individual models. Test dataset can be used to test GM also.
        """
        train_set = self.get_train_dataset()
        val_set = self.get_val_dataset()
        test_set = self.get_evaluation_set()
        return train_set, val_set, test_set

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"input timestep: {self.input_width}",
                f"output timestep: {self.label_width}",
            ]
        )
>>>>>>> 1baa0665dde96b77262a05f4b9715194a810c58e
