from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_learning_curve(history, folders: List[str]):
    # Do nothing if train stop manually
    if history is None:
        print("No train history was found")
        return
    else:
        # Create the plot
        plt.figure("Learning curve")
        plt.plot(
            history.epoch,
            history.history["loss"],
            history.epoch,
            history.history["val_loss"],
        )
        plt.legend(["train loss", "val loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.draw()
        # Save according on the desired directories
        for folder in folders:
            plt.savefig(folder)
        return


def plot_results(
    y_true, y_pred, out_labels, R2_score, rmse_result, max_error, nrmse, folders: list
):
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure("Prediction", figsize=(7, 4))
    time = [i / 20 for i in range(len(y_true))]
    for i, col in enumerate(list(out_labels)):
        plt.subplot(len(out_labels), 1, i + 1)
        print(f"{col} R2 score: {R2_score[i]}")
        print(f"{col} RMSE result: {rmse_result[i]}")
        print(f"{col} max error is {max_error[i]}Nm/Kg")
        print(f"{col} nRMSE is {nrmse[i]}Nm/Kg")

        plt.plot(time, y_true[:, i], "b-", linewidth=1.5)
        plt.plot(
            time,
            y_pred[:, i],
            "r--",
            linewidth=1,
        )
        plt.title(col, fontsize=title_size)
        if i == 0:
            plt.legend(["Measurments", "Estimations"], fontsize=label_size)
        plt.xlim((0, 9))
        plt.xlabel("Time [s]", fontsize=label_size)
        if "moment" in col:
            plt.ylabel(f"{col} [Nm/kg]", fontsize=label_size)
        else:
            plt.ylabel(f"{col} [Degree]", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.grid(True)
        plt.axhline(y=0, color="black", linestyle="-")
    plt.tight_layout()
    for folder in folders:
        plt.savefig(folder)
    plt.draw()


def plot_models(
    predictions: dict, y_true, labels: List[str], subject: str, folders: List[str]
):
    """
    This function will plot make plots to compare algorithm performance on a certain subject
    """
    from matplotlib import rcParams

    rcParams["ps.fonttype"] = 42
    time = [i / 20 for i in range(len(y_true))]
    # fig, ax = plt.subplots(nrows=len(predictions.keys()))
    tick_size = 12
    label_size = 14
    title_size = 20
    for j, label in enumerate(labels):
        plt.figure(f"S{subject} {label}", figsize=(11, 9))
        for i, model_name in enumerate(predictions.keys()):
            # Create subplots
            plt.subplot(len(predictions.keys()), 1, i + 1)
            plt.plot(time, y_true[:, j], "b", linewidth=2, label="measured moment")
            plt.plot(
                time,
                predictions[model_name][:, j],
                "r--",
                linewidth=1.5,
                label="prediction",
            )
            plt.title(model_name, fontsize=title_size)
            plt.xlim((0, 9))
            plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
            # plt.yticks([0, 0.5, 1, 1.5], fontsize=tick_size)
            plt.ylim([-1.6, 0.25])
            if "knee" in label.lower():
                plt.ylim([-0.4, 0.75])

            plt.grid(True)
        plt.xticks(fontsize=tick_size)
        plt.xlabel("Time [s]", fontsize=label_size)
        plt.legend(
            bbox_to_anchor=(1, -0.5),
            loc="lower right",
            borderaxespad=0,
            ncol=2,
            fontsize=label_size,
        )
        plt.tight_layout()
        plt.draw()
        for path in folders:
            plt.savefig(path)


def plot_data_only(
    y_true: np.array,
    y_pred: np.array,
    label: str,
    subject: str,
    folders: List[str],
    number_of_plots=3,
):
    """
    This function will plot n period estimations vs ground truth moment
    according to the number_of_plots and save the results in the desired
    folders.
    """
    # Create a pandas df to work easily
    df = pd.DataFrame({"measured": y_true[:, 0], "pred": y_pred[:, 0]})
    # Create the time column and set it as the index
    df.index = df.index / 20
    # First get the periods.
    periods = []
    start_ = 0
    i = 1
    while i < len(df):
        if len(periods) >= number_of_plots:
            break

        is_nan = df.iloc[i, :].isnull().values.any()
        if (not is_nan and df.iloc[i - 1, :].isnull().values.any()) or (
            i == 1 and not is_nan
        ):
            start_ = i
            i += 1
            while not df.iloc[i, :].isnull().values.any():
                i += 1
            periods.append(slice(start_, i - 1))
        else:
            i += 1
    # create subplots
    _, axes = plt.subplots(1, number_of_plots, figsize=(12, 5), sharey=True)
    for i in range(number_of_plots):
        axes[i].plot(df.iloc[periods[i], 0], "b", linewidth=2, label="measured moment")
        axes[i].plot(df.iloc[periods[i], 1], "r--", linewidth=1.5, label="prediction")
        axes[i].set_xticks([])
        axes[i].set_xlim([periods[i].start / 20, periods[i].stop / 20])
        # Set the y-axis limit according to the label
        if label == "ankle":
            axes[i].set_ylim([-1.7, 0.26])
        elif label == "knee":
            axes[i].set_ylim([-0.6, 0.6])
    axes[0].set_ylabel("Moment [Nm/kg]")
    plt.tight_layout()
    # Save the plot
    for folder in folders:
        plt.savefig(folder)
    plt.draw()
    return
