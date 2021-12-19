import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from Custom.models_functions import *
from Custom.WindowGenerator import WindowGenerator
from tensorflow.keras import backend as K


def create_window_generator(subject=None):
    if subject == None:
        subject = input("Please input subject number in XX format: ")
    if len(subject) == 1:
        subject = "0" + subject
    # #Get subject weight.
    w = subject_details[f"S{subject}"]["weight"]
    # #Get trials directory
    trials = ["train_01", "train_02", "val", "test"]
    trials = list(map(lambda x: f"../Dataset/S{subject}/{x}_dataset.csv", trials))
    # #Load data
    train_01_df = (pd.read_csv(trials[0], index_col="time"),)
    train_02_df = pd.read_csv(trials[1], index_col="time")
    val_df = pd.read_csv(trials[2], index_col="time")
    test_df = pd.read_csv(trials[3], index_col="time")
    # #Prepare scalers
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaler.fit(train_01_df.iloc[:400, :-8])
    angle_scaler = MinMaxScaler(feature_range=(0, 1))
    angle_scaler.fit(train_01_df.iloc[:400, -8:-4])
    # #Scale the dataset
    for data in [train_01_df, train_02_df, val_df, test_df]:
        data = scale_function(
            data, weights=w, features_scaler=features_scaler, angle_scaler=angle_scaler
        )

    train_01_df = choose_features(train_01_df, features=features)
    train_02_df = choose_features(train_02_df, features=features)
    val_df = choose_features(val_df, features=features)
    test_df = choose_features(test_df, features=features)
    # #Create Window object
    window_object = WindowGenerator(
        train_01_df,
        train_02_df,
        val_df,
        test_df,
        input_width=15,
        shift=3,
        label_width=1,
        batch_size=64,
        add_knee=add_knee,
        out_labels=out_labels,
    )
    return window_object


def train_fit_gm(
    subject,
    test_subject,
    model_name,
    epochs=1,
    lr=0.001,
    eval_only=False,
    load_best=False,
):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
    # setup model folder
    global folder
    folder = f"../Results/GM/{model_name}/S{test_subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_file = f"{folder}S{test_subject}_{model_name}.hdf5"
    # #FOR NOW I HAVE 2 SUBJECTS FOR TRAINING, I'LL GENERALIZE THE CODE TO ACCEPT MORE SUBJECTS LATER
    window_object_1 = create_window_generator(subject[0])
    window_object_2 = create_window_generator(subject[1])
    # Get all dataset
    train_set_1, val_set_1 = window_object_1.get_gm_train_val_dataset()
    train_set_2, val_set_2 = window_object_1.get_gm_train_val_dataset()
    train_set = window_object_1.preprocessing(train_set_1.concatenate(train_set_2))
    val_set = window_object_1.preprocessing(val_set_1.concatenate(val_set_2))

    ##############################################################################################################
    # Load and compile new model
    K.clear_session()
    model = model_dic[model_name](window_object_1)
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=lr), loss="mean_squared_error"
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{folder}S{test_subject}_{model_name}.hdf5",
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=True,
    )

    if load_best:
        try:
            model.load_weights(model_file)
        except:
            print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Train Model
    try:  # Train or load the best model the model
        if not eval_only:
            history = model.fit(
                x=train_set,
                validation_data=val_set,
                epochs=epochs,
                callbacks=[model_checkpoint_callback],
            )
            plot_learning_curve(history, folder)
            # Load the best model
            model.load_weights(model_file)
        else:
            history = "No training was conducted"
            model.load_weights(model_file)
    except KeyboardInterrupt:
        history = "\n\nTrains stopped manually"
        print(history)
    except OSError:  # If no saved model to be evaluated exist
        print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Get predictions and real values
    test_window = create_window_generator(test_subject)
    w = subject_details[f"S{test_subject}"]["weight"]
    test_set = test_window.get_evaluation_set()
    y_pred = model.predict(test_set)
    if len(y_pred.shape) == 3:
        # Get the last time step and reduce output dimenions to two
        y_pred = y_pred[:, -1, :]
    # Get real outputs
    for _, y_true in test_set.as_numpy_iterator():
        break
    y_true = y_true[:, -1, :]

    ################ Evaluation and plot ################
    r2_score = nan_R2(y_true, y_pred)
    rmse_result = nan_rmse(y_true, y_pred)
    plot_results(y_true, y_pred, out_labels, r2_score, rmse_result, folder)
    plt.draw()
    return history, y_true, y_pred, r2_score, rmse_result


if __name__ == "__main__":
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    features = ["RMS", "ZC"]
    add_knee = False
    out_labels = ["ankle moment"]
    model_name = "nn_model"

    model_dic = {}

    model_dic["lstm_model"] = create_lstm_model
    model_dic["conv_model"] = create_conv_model
    model_dic["nn_model"] = create_nn_model

for model_name in model_dic.keys():
    history, y_true, y_pred, r2, rmse = train_fit_gm(
        subject=["01", "02"],
        test_subject="04",
        model_name=model_name,
        epochs=1000,
        eval_only=False,
        load_best=False,
    )
    print(model_name)
