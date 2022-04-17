import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re
import os
import json


def load_data(trial_files):
    for side in trial_files:
        if "2.csv" in side:
            data_R = pd.read_csv(input_path+side, header=31)
        elif "1.csv" in side:
            data_L = pd.read_csv(input_path+side, header=31)
    return data_L, data_R


def remove_system_gap(data_L, data_R):
    """
    In some cases force plates stop recording and send only zeros. This function will\\
        first set these values for NaN and then interpolate missing values.
    """
    columns = data_L.columns[3:-1]
    data_L.loc[data_L[' Fy'] == 0, columns] = np.nan
    data_R.loc[data_R[' Fy'] == 0, columns] = np.nan
    data_L.iloc[:, :] = data_L.interpolate(method="linear")
    data_R.iloc[:, :] = data_R.interpolate(method="linear")
    data_L.iloc[:, :] = data_L.fillna(method="bfill")
    data_R.iloc[:, :] = data_R.fillna(method="bfill")
    return data_L, data_R


def remove_offset(data_L, data_R, remove=True):
    if remove:
        columns = data_L.columns[3:-3]  # Choose Forces and Moments
        for col in columns:
            data_L.loc[:, col] = data_L.loc[:, col] - \
                data_L.loc[5:60, col].mean()
            data_R.loc[:, col] = data_R.loc[:, col] - \
                data_R.loc[5:60, col].mean()
    return data_L, data_R


def grf_periods(data, trigger=5):
    stance_periods = []
    start = None
    for i, point in enumerate(data[" Fz"]):
        # set starting point if this is the first point
        if point >= trigger:
            if start == None:
                start = i
        elif point < trigger and start != None:
            # sometimes the subject might enter forceplate and leave it immediately because he enter with wrong foot. the filter will raise error if the number of points is less than 15
            if i-start >= 20:
                stance_periods.append(slice(start-3, i+3))
            start = None
    return stance_periods


def apply_filter(data, trigger=3):
    stance_periods = grf_periods(data, trigger)
    f = 5  # Filter frequency
    fs = 100  # Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=4, Wn=low_pass, btype='lowpass')
    columns = [" Fx", " Fy", " Fz",
               " Mx", " My", " Mz", ]
    # apply filter
    for stance in stance_periods:
        data.loc[stance, columns] = filtfilt(
            b2, a2, data.loc[stance, columns], axis=0)
        # CoP are calculated from the Force and Moment. Filter CoP by recalculate it from the filtered data. Note that the CoP will grow when foot outside force plate.
        data.loc[stance, " Cx"] = - \
            data.loc[stance, " My"]/data.loc[stance, " Fz"]
        data.loc[stance, " Cy"] = data.loc[stance, " Mx"] / \
            data.loc[stance, " Fz"]
    columns.extend([" Cx", " Cy", " Cz"])
    data.loc[0:stance_periods[0].start, columns] = 0
    for i in range(1, len(stance_periods)):
        previous_end = stance_periods[i-1].stop
        current_start = stance_periods[i].start
        data.loc[previous_end:current_start, columns] = 0
    return data


def system_match(data_L, data_R):
    """
    Match opti-track and force plates axes.
    """
    # To apply rotation, change column names.
    col_names = {" Fx": "Fx", " Fy": "Fz", " Fz": "Fy",
                 " Mx": "Mx", " My": "Mz", " Mz": "My",
                 " Cx": "Cx", " Cy": "Cz", " Cz": "Cy",
                 " MocapTime": "time"}

    data_L.rename(columns=col_names, inplace=True)
    data_R.rename(columns=col_names, inplace=True)

    # Match opti-track and force Plates origins
    data_L.loc[:, "Cz"] = data_L["Cz"] + 0.25
    data_R.loc[:, "Cz"] = data_R["Cz"] + 0.25
    data_L.loc[:, "Cx"] = data_L["Cx"] + 0.25
    data_R.loc[:, "Cx"] = data_R["Cx"] + 0.75
    # Complete the rotation by getting -z
    change_sign = ["Fx", "Fz"]
    data_L.loc[:, change_sign] = -data_L[change_sign]
    data_R.loc[:, change_sign] = -data_R[change_sign]
    return data_L, data_R


# There is a delay in system (various delay may change )
def shift_data(data_L, data_R, shift_key):
    shift_columns = [' Fx', ' Fz', ' Fy',
                     ' Mx', ' Mz', ' My', ]
    left_shift = subject_details[f"S{subject}"]["delay"][shift_key][0]
    right_shift = subject_details[f"S{subject}"]["delay"][shift_key][1]
    data_L.loc[:, shift_columns] = data_L[shift_columns].shift(
        left_shift, fill_value=0)
    data_R.loc[:, shift_columns] = data_R[shift_columns].shift(
        right_shift, fill_value=0)
    return data_L, data_R


def force_plate_pipeline(data_L, data_R):
    # System stop working someat some frames creating a gap, fill the gaps using interpolatoion
    data_L, data_R = remove_system_gap(data_L, data_R)
    # Remove the delay
    data_L, data_R = shift_data(data_L, data_R, shift_key=trials[i])
    # Remove the offset from the data
    data_L, data_R = remove_offset(data_L, data_R)
    # Filter the data
    data_L = apply_filter(data_L)
    data_R = apply_filter(data_R)
    # Match devices coordinate system
    data_L, data_R = system_match(data_L, data_R)
    # Make sure delta time is 0.01
    data_L['time'] = data_L["MocapFrame"]/100
    data_R['time'] = data_R["MocapFrame"]/100
    return data_L, data_R


def col_rearrange(data):
    return data[["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cx", "Cy", "Cz"]]


# Rename columns for OpenSim
def GRF_data(data_L, data_R):
    data_L = col_rearrange(data_L)
    data_R = col_rearrange(data_R)
    R_columns_names_mapper = {"Fx": "ground_force_vx", "Fy": "ground_force_vy", "Fz": "ground_force_vz",
                              "Cx": "ground_force_px", "Cy": "ground_force_py", "Cz": "ground_force_pz",
                              "Mx": "ground_torque_x", "My": "ground_torque_y", "Mz": "ground_torque_z"}

    L_columns_names_mapper = {
        key: f"1_{R_columns_names_mapper[key]}" for key in R_columns_names_mapper}
        
    data_L.rename(columns=L_columns_names_mapper, inplace=True)
    data_R.rename(columns=R_columns_names_mapper, inplace=True)

    GRF = pd.merge(left=data_L, right=data_R, how="outer", on="time")
    return GRF


def save_force_data(force_data, output_path, output_name):
    output_file = output_path + output_name
    if os.path.exists(output_file):
        os.remove(output_file)
    force_data.to_csv(output_file,  sep='\t', index=False)
    nRows = len(force_data)  # end_time - start_time + 1
    nColumns = 19

    with open(output_file, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(output_name+'\n' + 'version=1\n' +
                f'nRows={nRows}\n' + f'nColumns={nColumns}\n' + 'inDegrees=yes\n' + 'endheader\n' + old)


if __name__ == "__main__":
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        
    subject = "02"#input(f"insert subject number in XX format: ")
    date = subject_details[f"S{subject}"]["date"]
    input_path = f"../Data/S{subject}/{date}/Dynamics/Force_Data/"
    output_path = f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data/"
    # Get files names
    trials = ["train_01", "train_02", "val", "test"]
    files = [[f"S{subject}_{trial}_forceplate_{i}.csv" for i in [1, 2]]
             for trial in trials]
    # Process each trial
    for i, trial_files in enumerate(files):
        # Load Left(1) and Right(2) force plates data
        data_L, data_R = load_data(trial_files)
        data_L, data_R = force_plate_pipeline(data_L, data_R)
        force_data = GRF_data(data_L, data_R)
        # Save force data
        output_name = re.sub(
            "_forceplate_[0-9].csv", "_grf.sto", trial_files[0])
        save_force_data(force_data, output_path, output_name)
