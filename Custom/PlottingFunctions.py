import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(history, folder):
    if history == None:
        print("No train history was found")
    else:
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
        plt.savefig(f"{folder}learning_curve.pdf")


def plot_results(y_true, y_pred, out_labels, R2_score, rmse_result, max_error, nrmse, folder):
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
        
        plt.plot(time, y_true[:, i], "b-", linewidth=2.5)
        plt.plot(time, y_pred[:, i], "r--", linewidth=2,)
        plt.title(col, fontsize=title_size)
        if i == 0:
            plt.legend(["measured moment", "prediction"], fontsize=label_size)
        plt.xlim((0, 9))
        plt.xlabel("Time [s]", fontsize=label_size)
        if "moment" in col:
            plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
        else:
            plt.ylabel("Angle [Degree]", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='-')
    plt.tight_layout()
    plt.savefig(f"{folder}Predictions.pdf")
    plt.savefig(f"{folder}Predictions.svg")
    plt.draw()


def plot_models(predictions: dict, y_true, labels, subject, path: str):
    from matplotlib import rcParams

    rcParams['ps.fonttype'] = 42
    time = [i / 20 for i in range(len(y_true))]
    # fig, ax = plt.subplots(nrows=len(predictions.keys()))
    tick_size = 12
    label_size = 14
    title_size = 20
    for j, label in enumerate(labels):
        plt.figure(f"S{subject} {label}", figsize=(11, 9))
        for i, model_name in enumerate(predictions.keys()):
            # Create subplots
            plt.subplot(len(predictions.keys()), 1, i+1)
            plt.plot(time, y_true[:, j], "b", linewidth=2.5, label="measured moment")
            plt.plot(time, predictions[model_name][:, j], "r--", linewidth=2, label="prediction")
            plt.title(model_name, fontsize=title_size)
            plt.xlim((0, 9))
            plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
            # plt.yticks([0, 0.5, 1, 1.5], fontsize=tick_size)
            plt.ylim([-1.6, 0.25])
            if 'knee' in label.lower():
                plt.ylim([-0.4, 0.75])

            plt.grid(True)
        plt.xticks(fontsize=tick_size)
        plt.xlabel("Time [s]", fontsize=label_size)
        plt.legend(bbox_to_anchor=(1, -0.5), loc="lower right",
                borderaxespad=0, ncol=2, fontsize=label_size)
        plt.tight_layout()
        plt.draw()
        plt.savefig(f"{path}S{subject} {label} estimations.pdf")
        plt.savefig(f"{path}S{subject} {label} estimations.svg")
        
        
def plot_data_only(y_true, y_pred, label, subject, path: str, number_of_plots=3):
    df = pd.DataFrame({"measured" : y_true[:,0], "pred" : y_pred[:,0]})
    df.index = df.index/20
    periods = []
    start_ = 0
    i = 1
    while i < len(df):
        
        if len(periods)>=number_of_plots:
            break
        
        is_nan = df.iloc[i,:].isnull().values.any()
        if (not is_nan and df.iloc[i-1,:].isnull().values.any()) or (i==1 and not is_nan):
            start_ = i
            i += 1
            while not df.iloc[i,:].isnull().values.any():
                i += 1
            periods.append(slice(start_ , i-1))
        else:
            i += 1
    tick_size = 12
    label_size = 14
    title_size = 20
    fig , axes = plt.subplots(1, number_of_plots, figsize=(12, 5), sharey=True)    
    for i in range(number_of_plots):
        axes[i].plot(df.iloc[periods[i], 0], "b", linewidth=2.5, label="measured moment")
        axes[i].plot(df.iloc[periods[i], 1], "r--", linewidth=2, label="prediction")
        axes[i].set_xticks([])
        axes[i].set_xlim([periods[i].start/20, periods[i].stop/20])
        if label=='ankle':
            axes[i].set_ylim([-1.7, 0.26])
        elif label=='knee':
            axes[i].set_ylim([-0.6, 0.6])
    axes[0].set_ylabel('Moment [Nm/kg]')
    if subject!='13' or label=='knee':
        plt.legend(bbox_to_anchor=(1, -0.03), loc="upper right",
                    borderaxespad=0, ncol=2, fontsize=label_size)
    plt.tight_layout()
    plt.savefig(f"{path}S{subject} {label} estimation.pdf")
    plt.savefig(f"{path}S{subject} {label} estimation.svg")
    plt.close()
        
        