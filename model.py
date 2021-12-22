from Custom.WindowGenerator import WindowGenerator
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tensorflow import keras
from Custom.models_functions import *

K = keras.backend
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
gpu_index = -1
tf.config.experimental.set_visible_devices(
    devices=gpus[gpu_index], device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[gpu_index], \
# [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


def train_fit(
    subject, tested_on, model_name, epochs=1, lr=0.001, eval_only=False, load_best=False
):
    # setup results and models folder
    folder = f"../Results/indiviuals/{model_name}/S{subject}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_file = f"{folder}S{subject}_{model_name}.hdf5"

    window_object = window_generator(subject)
    if tested_on == None:
        tested_on = subject

    w = subject_details[f"S{subject}"]["weight"]
    # Get all dataset
    train_set, val_set, _ = window_object.make_dataset()
    ##############################################################################################################
    # Load and compile new model
    K.clear_session()
    model = model_dic[model_name](window_object)
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=lr), loss=SPLoss(loss_factor)
    )
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
    window_object = window_generator(tested_on)
    _, _, test_set = window_object.make_dataset()
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
    # Change the folder to the test subject folder after loading the model
    folder = f"../Results/indiviuals/{model_name}/S{tested_on}/"
    plot_results(y_true, y_pred, out_labels, r2_score, rmse_result, folder)
    plt.draw()
    return history, y_true, y_pred, r2_score, rmse_result


if __name__ == "__main__":
    if not tf.test.is_built_with_cuda():
        raise print("No GPU found")

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    # Choose features and labels
    features = ["RMS", "ZC"]  # Used EMG features
    add_knee = False  # True if you want to use knee angle as an extra input
    out_labels = ["ankle moment"]  # Labels to be predicted
    loss_factor = 5.0  # Loss factor to prevent ankle slip
    # Window object parameters
    input_width = 20
    shift = 3
    label_width = 1
    batch_size = 64

    window_generator = partial(create_window_generator, input_width=input_width, shift=shift, label_width=label_width,
                               batch_size=batch_size, features=features, add_knee=add_knee, out_labels=out_labels)

    model_dic = {}

    model_dic["lstm_model"] = create_lstm_model
    model_dic["conv_model"] = create_conv_model
    model_dic["nn_model"] = create_nn_model

    # w1 = window_generator(subject="1")
    # w2 = window_generator(subject="2")
    # w4 = window_generator(subject="4")

    # Train and test new/existing models
    for model_name in model_dic.keys():
        history, y_true, y_pred, r2, rmse = train_fit(
            subject="02",
            tested_on=None,
            model_name=model_name,
            epochs=1000,
            eval_only=False,
            load_best=False,
        )
        print(model_name)
    plt.show()
