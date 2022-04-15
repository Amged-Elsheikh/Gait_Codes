import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import re
import os
import json

# Rename columns for OpenSim


def get_output_name(pair):
    return re.sub("_forceplate_[1|2].csv", "_grf.sto", pair)


def system_match(data):
    """
    Match opti-track and force plates axes.
    Remove system gaps if any.

    """
    # To apply rotation, change column names.
    col_names = {" Fx": "Fx", " Fy": "Fz", " Fz": "Fy",
                 " Mx": "Mx", " My": "Mz", " Mz": "My",
                 " Cx": "Cx", " Cy": "Cz", " Cz": "Cy",
                 " MocapTime": "time"}

    data.rename(columns=col_names, inplace=True)
    ## System stop working someat some frames creating a gap, fill the gaps using interpolatoion
    # data = remove_system_gap(data)
    # Match opti-track and force Plates origins
    data.loc[:, "Cx"] = data["Cx"].apply(lambda x: (x+0.25))
    data.loc[:, "Cz"] = data["Cz"].apply(lambda x: (x+0.25))
    # Complete the rotation
    change_sign = ["Fx","Fz"]
    data.loc[:, change_sign] = -data[change_sign]
    return data


def remove_system_gap(data):
    """
    In some cases force plates stop recording and send only zeros. This function will\\
        first set these values for NaN and then interpolate missing values.
    """
    columns = data.columns[3:]
    data.loc[data['Fy'] == 0, columns] = np.nan
    data.iloc[:, :] = data.interpolate(method="linear")
    data.iloc[:, :] = data.fillna(method="bfill")
    return data


def remove_offset(data, remove=True):
    if remove:
        # Choose Forces and Moments
        columns = [" Fx", " Fy", " Fz", " Mx", " My"," Mz"]
        for col in range(len(columns)):
            data.iloc[:, col] = data.iloc[:, col] - data.iloc[5:15, col].mean()
    return data


# There is a delay in system (various delay may change )
def shift_data(data, shift_key):
    shift_columns = data.columns[3:]
    shift_value = subject_details[f"S{subject}"]["delay"][shift_key][0]
    data.loc[:, shift_columns] = data[shift_columns].shift(
        shift_value, fill_value=0)
    return data


def apply_filter(data):
    f = 8  # Filter frequency
    fs = 100  # Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    # columns = [" Fx", " Fy", " Fz",
    #            " Mx", " My", " Mz",]
    # for col in columns:
    #     data.loc[:, col] = filtfilt(b2, a2, data.loc[:, col], axis=0)
    # # CoP are calculated from the Force and Moment. 
    # # Filter CoP by recalculate it from the filtered data. 
    # # Note that the CoP will grow when foot outside force plate.  
    # data.loc[:," Cx"] = -data.loc[:, " My"]/data.loc[:, " Fz"]
    # data.loc[:," Cy"] = data.loc[:, " Mx"]/data.loc[:, " Fz"]
    return data


def col_rearrange(data):
    return data[["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cx", "Cy", "Cz"]]


def GRF_data(data):
    data = col_rearrange(data)
    L_columns_names_mapper = {"Fx": "1_ground_force_vx",
                              "Fy": "1_ground_force_vy",
                              "Fz": "1_ground_force_vz",
                              "Cx": "1_ground_force_px",
                              "Cy": "1_ground_force_py",
                              "Cz": "1_ground_force_pz",
                              "Mx": "1_ground_torque_x",
                              "My": "1_ground_torque_y",
                              "Mz": "1_ground_torque_z"}

    data.rename(columns=L_columns_names_mapper, inplace=True)

    # GRF = pd.merge(left=data, right=data_R, how="outer", on="time")
    return data


def save_force_data(force_data, output_path, output_name):
    output_file = output_path + output_name
    if os.path.exists(output_file):
        os.remove(output_file)
    force_data.to_csv(output_file,  sep='\t', index=False)
    nRows = len(force_data)  # end_time - start_time + 1
    nColumns = 10

    with open(output_file, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(output_name+'\n' + 'version=1\n' +
                f'nRows={nRows}\n' + f'nColumns={nColumns}\n' + 'inDegrees=yes\n' + 'endheader\n' + old)


if __name__ == '__main__':

    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    subject = "06"  # input(f"insert subject number in XX format: ")
    date = subject_details[f"S{subject}"]["date"]
    w = subject_details[f"S{subject}"]["weight"]

    input_path = f"../Data/S{subject}/{date}/Dynamics/Force_Data/"
    output_path = f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data/"
    # Get files names
    trials = ["train_01", "train_02", "val", "test"]
    files = [f"S{subject}_{trial}_forceplate_1" for trial in trials]
    # Process each trial
    for i, file in enumerate(files):
        output_name = get_output_name(file)

        # Load Left force plates data
        data = pd.read_csv(input_path+file, header=31)
        # Remove the offset from the data
        data = remove_offset(data)
        # Apply low pass filter
        data = apply_filter(data)
        # Match devices coordinate system
        data = system_match(data)
        # Remove the delay
        data = shift_data(data, shift_key=trials[i])
        # data = filter_COP(data, "L")
        force_data = GRF_data(data)
        # Save force data
        save_force_data(force_data, output_path, output_name)
