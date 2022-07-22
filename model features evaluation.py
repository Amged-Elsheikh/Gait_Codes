import pandas as pd
from intrasubject_training import *


if __name__ == "__main__":

    select_GPU(3)
    # sensor_checker = '1+5+6+8'
    tf.random.set_seed(42)
    # True if you want to use knee angle as an extra input
    add_knee = False
    # Labels to be predicted
    joint = 'ankle'
    out_labels = [f"{joint} moment"]
    # Load data
    csv_file = f"{joint} features evaluation.csv"
    eval_df = pd.read_csv(f'../Results/{csv_file}',
                          index_col=[0, 1], header=[0, 1])
    sensors_list = []
    for col in eval_df.droplevel(1, 1).columns:
        if col not in sensors_list:
            sensors_list.append(col)
    models_dic = {}
    models_dic['RCNN'] = create_conv_model

    model_name = 'RCNN'
    for subject, features_string in eval_df.index:
        if subject not in [14]:
            continue
        test_subject = f"{subject:02d}"
        predictions = {}
        features = features_string.split("+")

        for sensors_id in sensors_list:
            sensors_num = sensors_id.split("+")
            sensors = [f'sensor {x}' for x in sensors_num]

            y_true, y_pred, r2, rmse, nrmse, models_compare_pdf, models_compare_svg =\
                train_fit(subject=test_subject,
                          tested_on=None,
                          model_name=model_name,
                          models_dic=models_dic,
                          epochs=1000,
                          lr=0.001,
                          eval_only=False,
                          load_best=False,
                          joint=joint,
                          input_width=20,
                          shift=1,
                          label_width=1,
                          batch_size=8,
                          features=features,
                          sensors=sensors,
                          add_knee=add_knee,
                          out_labels=out_labels)

            eval_df.loc[(int(test_subject), features_string),
                        (sensors_id, "R2")] = r2[0]
            eval_df.loc[(int(test_subject), features_string),
                        (sensors_id, "RMSE")] = rmse[0]
            eval_df.loc[(int(test_subject), features_string),
                        (sensors_id, "NRMSE")] = nrmse[0]
            eval_df.to_csv(f"../Results/3{csv_file}")

            # print(model_name)
            plt.close()
        plt.close()
