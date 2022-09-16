import re
from typing import *

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow import keras

from Custom.DataHandler import DataHandler
from Custom.OneSideWindowGenerator import *
from Custom.PlottingFunctions import *
from Custom.TFModelEvaluation import *
from Custom.TFModels import *

plt.style.use("ggplot")

MUSCLES = {"1": "RF", "2": "VM", "3": "VL", "4": "BF",
           "5": "ST", '6': "TA", '7': "SOL", '8': "GM",
           '9': "PB"}

# Loss factor to prevent ankle slip
LOSSES = {"ankle": 2, "knee": 1}


class SPLoss(losses.Loss):  # Slip Prevention Loss
    '''This custom loss will multiply the loss by a specific threshold when the true value is positive'''

    def __init__(self, threshold=3.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        if self.threshold == 0:
            self.threshold = 1

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_positive = y_true > 0
        squared_loss = tf.square(error) / 2
        squared_loss_for_positive = tf.math.scalar_mul(
            self.threshold, squared_loss)
        return tf.where(is_positive, squared_loss_for_positive, squared_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


def train_fit(subject: str, tested_on: Union[str, None], model_name: str, models_dic: Dict,
              epochs=1, lr=0.001, eval_only=False, load_best=False, joint='ankle', input_width=20,
              shift=1, label_width=1, batch_size=8, features: List[str] = ['RMS'], sensors: List[str] = ['sensor 1'],
              add_knee=False, out_labels: List[str] = [f"ankle moment"], emg_type='DEMG'):

    ################################## Get output directories ##################################
    model_file, models_compare_pdf, models_compare_svg,\
        predictions_pdf, predictions_svg,\
        learning_curve_pdf, learning_curve_svg = get_files(features, sensors,
                                                           joint, subject,
                                                           model_name, emg_type,
                                                           is_general_model=False)

    ######### Create Window Object and load trainining and Validation sets #########
    partial_window_generator_func = partial(create_window_generator,
                                            input_width=input_width, shift=shift,
                                            label_width=label_width, batch_size=batch_size,
                                            features=features, sensors=sensors, add_knee=add_knee,
                                            out_labels=out_labels, emg_type=emg_type, is_general_model=False)

    window_object = partial_window_generator_func(subject)
    train_set, val_set, _ = window_object.make_dataset()

    ################################ Train or Load Model ################################
    loss_factor = LOSSES[joint]
    model = compile_and_train(window_object, model_name,
                              model_file, models_dic,
                              train_set, val_set,
                              loss_factor, epochs,
                              lr, eval_only, load_best,
                              learning_curve_pdf, learning_curve_svg,)

    ################################## Make Estimations ##################################
    if tested_on == None:
        tested_on = subject
    y_true, y_pred = get_estimations(
        tested_on, partial_window_generator_func, model)

    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation_and_plot(y_true, y_pred,
                                                       tested_on, out_labels,
                                                       predictions_pdf, predictions_svg)
    return r2_score, rmse_result, nrmse, y_true, y_pred, models_compare_pdf, models_compare_svg


def get_files(features: List[str], sensors: List[str],
              joint: str, subject: str, model_name: str,
              emg_type='DEMG', is_general_model=False) -> List[str]:
    '''
    Get Outputs directories
    '''
    used_features = '+'.join(features)
    used_muscles = []
    for sensor in sensors:
        used_muscles.append(MUSCLES[re.search('[0-9]+', sensor)[0]])
    used_muscles = '+'.join(used_muscles)
    # save Intrasubject and Intersubject models seperatelt
    if is_general_model:
        parent_folder = f"../Results/GM/{emg_type}"
    else:
        parent_folder = f"../Results/indiviuals/{emg_type}"
    # Files name should be unique to each case, especially when saving the models.
    file_name = f'{joint}/S{subject} {joint} {model_name} {used_muscles} {used_features}'
    models_compare_file = f'{joint}/S{subject} {joint} {used_muscles} {used_features}'
    # Get the model and plots folder
    model_file = f"{parent_folder}/Models/{file_name}.hdf5"
    models_compare_pdf = f"{parent_folder}/Models compare/PDF/{models_compare_file} models compare.pdf"
    models_compare_svg = f"{parent_folder}/Models compare/SVG/{models_compare_file} models compare.svg"
    predictions_pdf = f"{parent_folder}/Predictions/PDF/{file_name} predictions.pdf"
    predictions_svg = f"{parent_folder}/Predictions/SVG/{file_name} predictions.svg"
    learning_curve_pdf = f"{parent_folder}/Learning Curve/PDF/{file_name} learning curve.pdf"
    learning_curve_svg = f"{parent_folder}/Learning Curve/SVG/{file_name} learning curve.svg"

    return [model_file, models_compare_pdf, models_compare_svg,
            predictions_pdf, predictions_svg,
            learning_curve_pdf, learning_curve_svg]


# # Window generator creation function
def create_window_generator(subject: str, input_width=20, shift=3,
                            label_width=1, batch_size=64, features=["RMS"],
                            sensors=['sensor 1'], add_knee=False,
                            out_labels=["ankle moment"], emg_type='sEMG',
                            is_general_model=False) -> WindowGenerator:
    '''
    A function what will handle creating the sliding window object
    '''
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    # Get scaled dataset.
    dataHandler = DataHandler(
        subject, features, sensors, add_knee, out_labels, emg_type)
    # # Create Window object
    window_object = WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size,
                                    is_general_model)
    return window_object


def compile_and_train(window_object: WindowGenerator, model_name: str,
                      model_file: str, models_dic: Dict,
                      train_set: tf.data.Dataset, val_set: tf.data.Dataset,
                      loss_factor: float = 1, epochs=1, lr=0.001,
                      eval_only=False, load_best=False,
                      learning_curve_pdf: Union[str, None] = None,
                      learning_curve_svg: Union[str, None] = None):
    '''This function is responsible for creating and training the model'''
    # Make sure no any cached data are stored
    keras.backend.clear_session()
    # Create the model.
    model = models_dic[model_name](window_object)
    # compile the model using NADAM compiler and a custom Loss function
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr),
                  loss=SPLoss(loss_factor))
    # Set the callbacks
    callbacks = model_callbacks(model_file)
    ############################################################################
    # Loading best model if user specified
    if load_best:
        try:
            model.load_weights(model_file)
        except OSError:
            print("No saved model existing. weights will be initialized")

    try:
        if not eval_only:
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=callbacks)
            if learning_curve_pdf and learning_curve_svg:
                plot_learning_curve(history,
                                    [learning_curve_pdf,
                                     learning_curve_svg])
            plt.close()

    # To stop the training manually, click Ctrl+C
    except KeyboardInterrupt:
        print("\n\nTrains stopped manually")
    # If there is no trained model to be evaluated, create one

    except OSError:
        print("\n\n No saved model existing. New model will be trained")
        eval_only = False
        load_best = False
        model = compile_and_train(window_object, model_name,
                                  model_file, models_dic, train_set,
                                  val_set, loss_factor, epochs,
                                  lr, eval_only, load_best,
                                  learning_curve_pdf,
                                  learning_curve_svg)
    # Load the best model
    model.load_weights(model_file)
    return model


def get_estimations(tested_on: str, window_generator_func: partial[WindowGenerator], model: tf.keras.Model) -> Tuple[np.array, np.array]:
    # Create test subject's window object
    window_object: WindowGenerator = window_generator_func(tested_on)
    # Get the evaluation set
    test_set = window_object.evaluation_set
    # Get the predictions
    y_pred = model.predict(test_set)
    # predict will return a 3d array
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs from the testset
    for _, y_true in test_set.as_numpy_iterator():
        break
    # make it a 2D vector
    y_true = y_true[:, -1, :]
    return y_true, y_pred


def evaluation_and_plot(y_true: np.array, y_pred: np.array,
                        tested_on: str, out_labels: List[str],
                        predictions_pdf: str, predictions_svg: str,
                        ) -> Tuple(List[Union[float, np.ndarray, Any]]):
    # Calculate the evaluation metrices
    r2_score = nan_R2(y_true, y_pred)
    rmse_result, nrmse = nan_rmse(y_true, y_pred)
    # Create plots
    for label in out_labels:
        plot_data_only(y_true=y_true, y_pred=y_pred,
                       label=label, subject=tested_on,
                       folders=[predictions_pdf, predictions_svg], number_of_plots=5)
    return r2_score, rmse_result, nrmse


def train_fit_gm(subject: List[str], tested_on: Union[str, None], model_name: str, models_dic: Dict,
                 epochs=1, lr=0.001, eval_only=False, load_best=False, joint='ankle', input_width=20,
                 shift=1, label_width=1, batch_size=8, features: List[str] = ['RMS'], sensors: List[str] = ['sensor 1'],
                 add_knee=False, out_labels: List[str] = [f"ankle moment"], emg_type='DEMG'):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
   ################################## Get Files ##################################
    model_file,\
        models_compare_pdf, models_compare_svg,\
        predictions_pdf, predictions_svg,\
        learning_curve_pdf, learning_curve_svg = get_files(features, sensors,
                                                           joint, tested_on,
                                                           model_name, emg_type,
                                                           is_general_model=True)

    ######### Create Window Object and load trainining and Validation sets ########
    window_generator = partial(create_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width, batch_size=batch_size,
                               features=features, sensors=sensors, add_knee=add_knee,
                               out_labels=out_labels, emg_type=emg_type, is_general_model=True)
    # Make dataset

    flag = True
    for s in subject:
        if flag:
            flag = False
            window_object = window_generator(s)
            train_set = window_object.train_dataset
            val_set = window_object.val_dataset
        else:
            window_object = window_generator(s)
            temp_train = window_object.train_dataset
            temp_val = window_object.val_dataset

            train_set = train_set.concatenate(temp_train)
            val_set = val_set.concatenate(temp_val)

    train_set = WindowGenerator.preprocessing(train_set, remove_nan=True,
                                              shuffle=True, batch_size=batch_size,
                                              drop_reminder=True,)

    val_set = WindowGenerator.preprocessing(val_set, remove_nan=True,
                                            shuffle=False, batch_size=10**6,
                                            drop_reminder=False,)

    ####################### Train or Load Model ###################################
    loss_factor = LOSSES[joint]
    model = compile_and_train(window_object, model_name,
                              model_file, models_dic,
                              train_set, val_set,
                              loss_factor, epochs,
                              lr, eval_only, load_best,
                              learning_curve_pdf, learning_curve_svg,)

    ######################### Make Estimations #####################################
    if tested_on == None:
        tested_on = subject
    y_true, y_pred = get_estimations(
        tested_on, window_generator, model)

    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation_and_plot(y_true, y_pred,
                                                       tested_on, out_labels,
                                                       predictions_pdf, predictions_svg)
    return r2_score, rmse_result, nrmse, y_true, y_pred, models_compare_pdf, models_compare_svg


def add_mean_std(df: pd.Series):
    '''Add mean and Std rows to a pandas sereis'''
    mean = df.mean()
    std = df.std()
    df.loc['mean', :] = mean
    df.loc['std', :] = std


def custom_loss(y_true, y_pred):
    error = tf.reshape(y_true - y_pred, (-1, 1))
    error = error[~tf.math.is_nan(error)]
    return tf.reduce_mean(tf.square(error), axis=0)
