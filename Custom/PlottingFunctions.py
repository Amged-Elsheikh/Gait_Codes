import matplotlib.pyplot as plt


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


def plot_results(y_true, y_pred, out_labels, R2_score, rmse_result, max_error, folder):
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure("Prediction", figsize=(11, 9))
    time = [i / 20 for i in range(len(y_true))]
    for i, col in enumerate(list(out_labels)):
        plt.subplot(len(out_labels), 1, i + 1)
        print(f"{col} R2 score: {R2_score[i]}")
        print(f"{col} RMSE result: {rmse_result[i]}")
        print(f"{col} max error is {max_error}Nm/Kg")
        plt.plot(time, -y_true[:, i], linewidth=2.5)
        plt.plot(time, -y_pred[:, i], "r--", linewidth=2,)
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
    plt.tight_layout()
    plt.savefig(f"{folder}{out_labels[i]}.svg")
    plt.savefig(f"{folder}{out_labels[i]}.pdf")
    plt.draw()


def plot_models(predictions: dict, y_true, path: str, subject=None):
    from matplotlib import rcParams

    rcParams['ps.fonttype'] = 42
    time = [i / 20 for i in range(len(y_true))]
    # fig, ax = plt.subplots(nrows=len(predictions.keys()))
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure(f"S{subject} GM", figsize=(11, 9))
    for i, model_name in enumerate(predictions.keys()):
        plt.subplot(len(predictions.keys()), 1, i+1)
        plt.plot(time, -y_true, linewidth=2.5, label="measured moment")
        plt.plot(time, -predictions[model_name],
                 "r--", linewidth=2, label="prediction")
        plt.title(model_name, fontsize=title_size)
        plt.xlim((0, 9))
        plt.ylabel("Moment [Nm/kg]", fontsize=label_size)
        plt.yticks([0, 0.5, 1, 1.5], fontsize=tick_size)
        plt.ylim([-0.25, 1.52])
        plt.grid(True)
        plt.axvspan(3, 5.55, alpha=0.3, color='blue')
    plt.xticks(fontsize=tick_size)
    plt.xlabel("Time [s]", fontsize=label_size)
    plt.legend(bbox_to_anchor=(1, -0.5), loc="lower right",
               borderaxespad=0, ncol=2, fontsize=label_size)
    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{path}S{subject}_models_results.pdf")
