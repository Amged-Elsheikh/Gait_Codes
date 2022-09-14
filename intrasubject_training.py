import re
from typing import *

import matplotlib.pyplot as plt
import tensorflow as tf
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


def add_mean_std(df):
    mean = df.mean()
    std = df.std()
    df.loc['mean', :] = mean
    df.loc['std', :] = std


def get_files(features: List[str], sensors: List[str], joint: str, subject: str,
 model_name: str, emg_type='DEMG', is_general_model=False):
    used_features = '+'.join(features)
    used_muscles = []
    for sensor in sensors:
        used_muscles.append(MUSCLES[re.search('[0-9]+', sensor)[0]])
    used_muscles = '+'.join(used_muscles)
    if is_general_model:
        folder = f"../Results/GM/{emg_type}"
    else:
        folder = f"../Results/indiviuals/{emg_type}"
    file_name = f'{joint}/S{subject} {joint} {model_name} {used_muscles} {used_features}'
    models_compare_file = f'{joint}/S{subject} {joint} {used_muscles} {used_features}'

    model_file = f"{folder}/Models/{file_name}.hdf5"
    models_compare_pdf = f"{folder}/Models compare/PDF/{models_compare_file} models compare.pdf"
    models_compare_svg = f"{folder}/Models compare/SVG/{models_compare_file} models compare.svg"
    predictions_pdf = f"{folder}/Predictions/PDF/{file_name} predictions.pdf"
    predictions_svg = f"{folder}/Predictions/SVG/{file_name} predictions.svg"
    learning_curve_pdf = f"{folder}/Learning Curve/PDF/{file_name} learning curve.pdf"
    learning_curve_svg = f"{folder}/Learning Curve/SVG/{file_name} learning curve.svg"


    return model_file,\
        models_compare_pdf, models_compare_svg,\
        predictions_pdf, predictions_svg,\
        learning_curve_pdf, learning_curve_svg


# # Window generator creation function
def create_window_generator(subject=None, input_width=20, shift=3,
                            label_width=1, batch_size=64, features=["RMS"],
                            sensors=['sensor 1'], add_knee=False,
                            out_labels=["ankle moment"], emg_type='sEMG'
                            ) -> WindowGenerator:
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # Get subject weight.
    dataHandler = DataHandler(subject, features, sensors, add_knee, out_labels, emg_type)
    # #Create Window object
    window_object = WindowGenerator(dataHandler, input_width,
                                    label_width, shift, batch_size)
    return window_object


def model_callbacks(model_file):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file, save_weights_only=True,
        monitor="val_loss", save_best_only=True,)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                     min_delta=1e-3,
                                                     factor=0.7,  patience=20)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=50, restore_best_weights=True,)
    callbacks = [checkpoint_callback, reduce_lr, early_stop]
    return callbacks


def compile_and_train(model_name, model_file, models_dic,
                      train_set, val_set, loss_factor,
                      learning_curve_pdf, learning_curve_svg,
                      window_object, epochs=1, lr=0.001,
                      eval_only=False, load_best=False):

    keras.backend.clear_session()
    model = models_dic[model_name](window_object)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=lr),
                  loss=SPLoss(loss_factor))
    callbacks = model_callbacks(model_file)

    ############################################################################
    if load_best:
        try:
            model.load_weights(model_file)
        except:
            print("No saved model existing. weights will be initialized")
    # Train or load the best model the model
    try:
        if not eval_only:
            history = model.fit(x=train_set, validation_data=val_set,
                                epochs=epochs, callbacks=callbacks)
            plot_learning_curve(
                history, [learning_curve_pdf, learning_curve_svg])
            plt.close()
            # Load the best model
            print('loading weights')
            model.load_weights(model_file)
            print('Best weights')

        else:
            model.load_weights(model_file)
    # To stop the training manually, click Ctrl+C
    except KeyboardInterrupt:
        print("\n\nTrains stopped manually")
    # If there is no trained model to be evaluated.
    except OSError:
        print("\n\n No saved model existing. Model will be trained")
        
        model = compile_and_train(model_name, model_file, models_dic,
                    train_set, val_set, loss_factor,
                    learning_curve_pdf, learning_curve_svg,
                    window_object, epochs, lr,
                    eval_only=False, load_best=False)
    return model


def get_estimations(tested_on, window_generator, model):

    window_object = window_generator(tested_on)
    test_set = window_object.evaluation_set
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]
    return y_true, y_pred


def evaluation_and_plot(y_true, y_pred, tested_on, out_labels, predictions_pdf, predictions_svg):
    # Calculate the evaluation metrices
    r2_score = nan_R2(y_true, y_pred)
    rmse_result, nrmse, max_error = nan_rmse(y_true, y_pred)
    # Create plots
    for label in out_labels:
        plot_data_only(y_true=y_true, y_pred=y_pred,
                       label=label, subject=tested_on,
                       folders=[predictions_pdf, predictions_svg], number_of_plots=5)

    return r2_score, rmse_result, nrmse


def train_fit(subject, tested_on, model_name, models_dic,
              epochs=1, lr=0.001, eval_only=False,
              load_best=False, joint='ankle', input_width=20,
              shift=1, label_width=1, batch_size=8,
              features=['RMS'], sensors=['sensor 1'], add_knee=False,
              out_labels=[f"ankle moment"], emg_type='DEMG'):

    ################################## Get Files ##################################
    model_file,\
        models_compare_pdf, models_compare_svg,\
        predictions_pdf, predictions_svg,\
        learning_curve_pdf, learning_curve_svg = get_files(features, sensors,
                                                           joint, subject, model_name, emg_type)

    ######### Create Window Object and load trainining and Validation sets ########
    window_generator = partial(create_window_generator,
                               input_width=input_width, shift=shift,
                               label_width=label_width, batch_size=batch_size,
                               features=features, sensors=sensors,
                               add_knee=add_knee, out_labels=out_labels, emg_type=emg_type)

    window_object = window_generator(subject)
    train_set, val_set, _ = window_object.make_dataset()

    ####################### Train or Load Model ###################################
    loss_factor = LOSSES[joint]
    model = compile_and_train(model_name, model_file, models_dic,
                              train_set, val_set, loss_factor,
                              learning_curve_pdf, learning_curve_svg,
                              window_object, epochs,
                              lr, eval_only, load_best)

    ######################### Make Estimations #####################################
    if tested_on == None:
        tested_on = subject
    y_true, y_pred = get_estimations(tested_on, window_generator, model)

    ############################## Evaluation and plot ##############################
    r2_score, rmse_result, nrmse = evaluation_and_plot(y_true, y_pred,
                                                       tested_on, out_labels,
                                                       predictions_pdf, predictions_svg)
    return y_true, y_pred, r2_score, rmse_result, nrmse, models_compare_pdf, models_compare_svg


def select_GPU(gpu_index=0):
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    else:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

        tf.config.experimental.set_visible_devices(
            devices=gpus[gpu_index], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)


if __name__=='__main__':
    pass