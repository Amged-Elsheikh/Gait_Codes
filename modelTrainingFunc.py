import re
from typing import Dict, List, Union, Callable
from functools import partial, reduce

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses

from utilities.DataHandler import DataHandler
from utilities.OneSideWindowGenerator import WindowGenerator
from utilities.PlottingFunctions import plot_learning_curve
from utilities.TFModels import evaluate, model_callbacks

plt.style.use("ggplot")

MUSCLES = {
    "1": "RF",
    "2": "VM",
    "3": "VL",
    "4": "BF",
    "5": "ST",
    "6": "TA",
    "7": "SOL",
    "8": "GM",
    "9": "PB",
}


class SPLoss(losses.Loss):  # Slip Prevention Loss
    """This custom loss will multiply the loss by a
    specific threshold when the true value is positive"""

    def __init__(self, threshold=2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        if self.threshold == 0:
            self.threshold = 1

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_positive = y_true > 0
        squared_loss = tf.square(error) / 2
        positive_squared_loss = tf.math.scalar_mul(self.threshold, squared_loss)
        return tf.where(is_positive, positive_squared_loss, squared_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


def get_files(
    features: List[str],
    sensors: List[str],
    joint: str,
    subject: str,
    model_name: str,
    emg_type="DEMG",
    is_general_model=False,
):
    """
    Get Outputs directories
    """
    used_features = "+".join(features)
    used_muscles = "+".join(
        [MUSCLES[re.search("[0-9]+", sensor)[0]] for sensor in sensors]
    )
    if is_general_model:
        parent_folder = f"../Results/GM/{emg_type}"
    else:
        parent_folder = f"../Results/indiviuals/{emg_type}"
    file_name = (
        f"{joint}/S{subject} {joint} {model_name} {used_muscles} {used_features}"
    )
    model_file = f"{parent_folder}/Models/{file_name}.hdf5"
    predictions_pdf = f"{parent_folder}/Predictions/PDF/{file_name} predictions.pdf"
    predictions_svg = f"{parent_folder}/Predictions/SVG/{file_name} predictions.svg"
    learning_curve_pdf = (
        f"{parent_folder}/Learning Curve/PDF/{file_name} learning curve.pdf"
    )
    learning_curve_svg = (
        f"{parent_folder}/Learning Curve/SVG/{file_name} learning curve.svg"
    )
    return [
        model_file,
        predictions_pdf,
        predictions_svg,
        learning_curve_pdf,
        learning_curve_svg,
    ]


# # Window generator creation function
def create_window_generator(
    subject: str,
    input_width=20,
    shift=3,
    label_width=1,
    batch_size=64,
    features=["RMS"],
    sensors=["sensor 1"],
    label="ankle moment",
    emg_type="sEMG",
    is_general_model=False,
):
    """
    A function what will handle creating the sliding window object
    """
    try:
        subject = f"{int(subject):02d}"
    except Exception:
        raise "Subject variable should be a number"
    # Get scaled dataset.
    dataHandler = DataHandler(subject, features, sensors, label, emg_type)
    # # Create Window object
    window_object = WindowGenerator(
        dataHandler, input_width, label_width, shift, batch_size, is_general_model
    )
    return window_object


def compile_and_train(
    window_object: WindowGenerator,
    model_name: str,
    model_file: str,
    models_dict: Dict,
    train_set: tf.data.Dataset,
    val_set: tf.data.Dataset,
    epochs=1,
    lr=0.001,
    eval_only=False,
    load_best=False,
    learning_curve_pdf: str = None,
    learning_curve_svg: str = None,
):
    """This function is responsible for creating and training the model"""
    keras.backend.clear_session()
    model = models_dict[model_name](window_object)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr), loss=SPLoss(2))
    if load_best or eval_only:
        try:
            model.load_weights(model_file)
        except OSError:
            eval_only = load_best = False
            print("No saved model existing. weights will be initialized")
    try:
        if not eval_only:
            history = model.fit(
                x=train_set,
                validation_data=val_set,
                epochs=epochs,
                callbacks=model_callbacks(model_file),
            )
            plot_learning_curve(history, [learning_curve_pdf, learning_curve_svg])
            plt.close()
    # To stop the training manually, click Ctrl+C
    except KeyboardInterrupt:
        print("\n\nTrains stopped manually\n\n")
    # If there is no trained model to be evaluated, create one
    except OSError:
        print("\n\n No saved model existing. New model will be trained\n\n")
        eval_only = False
        load_best = False
        model = compile_and_train(
            window_object,
            model_name,
            model_file,
            models_dict,
            train_set,
            val_set,
            epochs,
            lr,
            eval_only,
            load_best,
            learning_curve_pdf,
            learning_curve_svg,
        )
    model.load_weights(model_file)
    return model


def get_estimations(
    tested_on: str, window_generator_func: Callable, model: tf.keras.Model
):
    window_object = window_generator_func(tested_on)
    test_set = window_object.evaluation_set
    y_pred = model.predict(test_set)
    y_pred = y_pred[:, -1, :]
    for _, y_true in test_set.as_numpy_iterator():
        y_true = y_true[:, -1, :]
        break
    return y_true, y_pred


def eval_and_plot(
    y_true: np.array,
    y_pred: np.array,
    tested_on: str,
    label: str,
    predictions_pdf: str,
    predictions_svg: str,
):
    r2_score, rmse_result, nrmse = evaluate(y_true, y_pred)
    # plot_data_only(
    #     y_true,
    #     y_pred,
    #     label,
    #     tested_on,
    #     folders=[predictions_pdf, predictions_svg],
    #     number_of_plots=5,
    # )
    return r2_score, rmse_result, nrmse


def add_mean_std(df: pd.Series):
    """Add mean and Std rows to a pandas sereis"""
    mean = df.mean()
    std = df.std()
    df.loc["mean", :] = mean
    df.loc["std", :] = std


def train_fit(
    subjects: Union[str, List[str]],
    tested_on: Union[str, None],
    model_name: str,
    models_dict: Dict,
    features: List[str] = ["RMS"],
    sensors: List[str] = ["sensor 1"],
    emg_type="sEMG",
    joint="ankle",
    is_general_model=False,
    *,
    eval_only=False,
    load_best=False,
    epochs=1,
    lr=0.001,
    input_width=20,
    shift=1,
    label_width=1,
    batch_size=8,
):
    label = f"{joint} moment"
    if tested_on is None:
        tested_on = subjects if isinstance(subjects, str) else subjects[0]
    if not is_general_model and isinstance(subjects, list):
        subjects = subjects[0]
    (
        model_file,
        predictions_pdf,
        predictions_svg,
        learning_curve_pdf,
        learning_curve_svg,
    ) = get_files(
        features, sensors, joint, tested_on, model_name, emg_type, is_general_model
    )
    window_generator = partial(
        create_window_generator,
        input_width=input_width,
        shift=shift,
        label_width=label_width,
        batch_size=batch_size,
        features=features,
        sensors=sensors,
        label=label,
        emg_type=emg_type,
        is_general_model=is_general_model,
    )
    if is_general_model:
        train_set = []
        val_set = []
        for s in subjects:
            window_object = window_generator(s)
            train_set.append(window_object.train_dataset)
            val_set.append(window_object.val_dataset)

        train_set = reduce(lambda x, y: x.concatenate(y), train_set)
        val_set = reduce(lambda x, y: x.concatenate(y), val_set)
        train_set = WindowGenerator.preprocessing(
            train_set,
            remove_nan=True,
            shuffle=True,
            batch_size=batch_size,
            drop_reminder=True,
        )
        val_set = WindowGenerator.preprocessing(
            val_set,
            remove_nan=True,
            shuffle=False,
            batch_size=10**6,
            drop_reminder=False,
        )
    else:
        window_object = window_generator(subjects)
        train_set, val_set, _ = window_object.make_dataset()

    model = compile_and_train(
        window_object,
        model_name,
        model_file,
        models_dict,
        train_set,
        val_set,
        epochs,
        lr,
        eval_only,
        load_best,
        learning_curve_pdf,
        learning_curve_svg,
    )
    y_true, y_pred = get_estimations(tested_on, window_generator, model)
    r2_score, rmse_result, nrmse = eval_and_plot(
        y_true, y_pred, tested_on, label, predictions_pdf, predictions_svg
    )
    return y_true, y_pred, r2_score, rmse_result, nrmse
