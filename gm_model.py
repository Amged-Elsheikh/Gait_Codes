import json
import os
from functools import partial
import pandas as pd
import tensorflow as tf
from Custom.models_functions import *

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
gpu_index = 1
tf.config.experimental.set_visible_devices(
    devices=gpus[gpu_index], device_type="GPU")


def train_fit_gm(subject, test_subject, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False,):
    """
    subject: List the subjects used for training.
    tested on: subject number in XX string format.
    """
    # #Create Results and model folder
    folder = f"../Results/GM/{model_name}/S{test_subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_file = f"{folder}S{test_subject}_{model_name}.hdf5"
    # Make dataset
    # #FOR NOW I HAVE 2 SUBJECTS FOR TRAINING, I'LL GENERALIZE THE CODE TO ACCEPT MORE SUBJECTS LATER
    window_object_1 = window_generator(subject[0])
    window_object_2 = window_generator(subject[1])
    # Get all dataset
    train_set_1, val_set_1 = window_object_1.get_gm_train_val_dataset()
    train_set_2, val_set_2 = window_object_2.get_gm_train_val_dataset()
    train_set = window_object_1.preprocessing(train_set_1.concatenate(train_set_2),
                                              remove_nan=True, shuffle=True, batch_size=None, drop_reminder=True,)

    val_set = window_object_1.preprocessing(val_set_1.concatenate(val_set_2),
                                            remove_nan=True, shuffle=False, batch_size=None, drop_reminder=False,
                                            )

    ##############################################################################################################
    # Load and compile new model
    tf.keras.backend.clear_session()
    model = model_dic[model_name](window_object_1)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr), loss=SPLoss(loss_factor)
    )
    # model.summary()
    # input("Click Enter to continue")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file,
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
        else:
            history = "No training was conducted"
    except KeyboardInterrupt:
        history = "\n\nTrains stopped manually"
        print(history)
    except OSError:  # If no saved model to be evaluated exist
        print("No saved model existing. weights will be initialized")
    ##############################################################################################################
    # Load the best model. Evaluation will always be with best model
    model.load_weights(model_file)
    # Get predictions and real values
    test_window = window_generator(test_subject)
    # w = subject_details[f"S{test_subject}"]["weight"]
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
    rmse_result, max_error = nan_rmse(y_true, y_pred)
    plot_results(y_true, y_pred, out_labels, r2_score,
                 rmse_result, max_error, folder)
    plt.close()
    return history, y_true, y_pred, r2_score, rmse_result


if __name__ == "__main__":
    tf.random.set_seed(42)

    # Check for GPU
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")
    # Get all subjects details
    # with open("subject_details.json", "r") as f:
    #     subject_details = json.load(f)
    # Choose features and labels
    features = ["RMS", "ZC"]  # Used EMG features
    add_knee = False  # True if you want to use knee angle as an extra input
    out_labels = ["ankle moment"]  # Labels to be predicted
    loss_factor = 3.0  # Loss factor to prevent ankle slip
    # Window object parameters
    input_width = 15
    shift = 3
    label_width = 1
    batch_size = 64

    window_generator = partial(create_window_generator, input_width=input_width, shift=shift, label_width=label_width,
                               batch_size=batch_size, features=features, add_knee=add_knee, out_labels=out_labels)
    # model_name = "nn_model"
    model_dic = {}

    model_dic["lstm_model"] = create_lstm_gm_model
    # model_dic["single_lstm_model"] = create_single_lstm_model
    model_dic["conv_model"] = create_conv_model
    model_dic["nn_model"] = create_nn_gm_model

    # Create pandas dataframe that will have all the results
    r2_results = pd.DataFrame(columns=model_dic.keys())
    rmse_results = pd.DataFrame(columns=model_dic.keys())
    test_subject = "04"
    train_subjects = ["01", "02"]
    for model_name in model_dic.keys():
        print(model_name)
        history, y_true, y_pred, r2, rmse = train_fit_gm(
            subject=train_subjects, test_subject=test_subject,
            model_name=model_name, epochs=1000,
            eval_only=False, load_best=False)

        r2_results.loc[f"S{test_subject}", model_name] = r2[0]
        rmse_results.loc[f"S{test_subject}", model_name] = rmse[0]
        plt.close()

    r2_results.to_csv("../Results/GM/R2_results.csv")
    rmse_results.to_csv("../Results/GM/RMSE_results.csv")
