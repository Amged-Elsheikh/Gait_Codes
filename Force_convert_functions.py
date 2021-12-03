import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re
import os
import json

with open("subject_details.json", "r") as f:
    subject_details = json.load(f)

subject = input(f"insert subject number in XX format: ")
date = subject_details[f"S{subject}"]["date"]
w = subject_details[f"S{subject}"]["weight"]


def read_data(path): 
    return pd.read_csv(path, header=31)
# Rename columns for OpenSim


def get_output_name(pair):
     return re.sub("_forceplate_[0-9].csv", "_grf.sto", pair)


def remove_system_gap(data_L, data_R):
    """
    In some cases force plates stop recording and send only zeros. This function will\\
        first set these values for NaN and then interpolate missing values.
    """
    columns = data_L.columns[3:-1]
    data_L.loc[data_L['Fy'] == 0, columns] = np.nan
    data_R.loc[data_R['Fy'] == 0, columns] = np.nan
    data_L.iloc[:,:] = data_L.interpolate(method="linear")
    data_R.iloc[:,:] = data_R.interpolate(method="linear")
    data_L.iloc[:,:] = data_L.fillna(method="bfill")
    data_R.iloc[:,:] = data_R.fillna(method="bfill")
    return data_L, data_R


def system_match(data_L, data_R):
    """
    Match opti-track and force plates axes.
    Remove system gaps if any.

    """
    # To apply rotation, change column names.
    col_names = {" Fx": "Fx", " Fy": "Fz", " Fz": "Fy",
                 " Mx": "Mx", " My": "Mz", " Mz": "My",
                 " Cx": "Cx", " Cy": "Cz", " Cz": "Cy",
                 " MocapTime": "time"}

    data_L.rename(columns=col_names, inplace=True)
    data_R.rename(columns=col_names, inplace=True)
    # System stop working someat some frames creating a gap, fill the gaps using interpolatoion
    data_L, data_R = remove_system_gap(data_L, data_R)
    # Match opti-track and force Plates origins
    data_L.loc[:, "Cz"] = data_L["Cz"].apply(lambda x: x+0.25)
    data_R.loc[:, "Cz"] = data_R["Cz"].apply(lambda x: x+0.25)
    data_L.loc[:, "Cx"] = data_L["Cx"].apply(lambda x: x+0.25)
    data_R.loc[:, "Cx"] = data_R["Cx"].apply(lambda x: x+0.75)
    # Complete the rotation by getting -z
    data_L.loc[:, ["Fz", "Fx", "Mx", "Mz"]] = data_L[[
        "Fz", "Fx", "Mx", "Mz"]].apply(lambda x: -x)
    data_R.loc[:, ["Fz", "Fx", "Mx", "Mz"]] = data_R[[
        "Fz", "Fx", "Mx", "Mz"]].apply(lambda x: -x)
    return data_L, data_R


def remove_offset(data_L, data_R, remove=True):
    if remove:
        columns = data_L.columns[3:-3]  # Choose Forces and Moments
        for col in columns:
            data_L.loc[:, col] = data_L.loc[:, col] - data_L.loc[5:60, col].mean()
            data_R.loc[:, col] = data_R.loc[:, col] - data_R.loc[5:60, col].mean()
    return data_L, data_R


f = 6  # Filter frequency
fs = 100  # Hz
low_pass = f/(fs/2)
b2, a2 = butter(N=2, Wn=low_pass, btype='lowpass')


def apply_filter(data_L, data_R):
    columns = ["Fx", "Fy", "Fz",
               "Mx", "My", "Mz",
               "Cx", "Cz"]
    for col in columns:
        data_L.loc[:, col] = filtfilt(b2, a2, data_L.loc[:, col])
        data_R.loc[:, col] = filtfilt(b2, a2, data_R.loc[:, col])
    # Make any force less than a 0.1*w to zero
    data_L.loc[np.abs(data_L['Fy']) < 0.1*w,
               ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']] = 0
    data_R.loc[np.abs(data_R['Fy']) < 0.1*w,
               ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']] = 0
    return data_L, data_R


# There is a delay in system (various delay may change )

def shift_data(data_L, data_R, shift_key):
    shift_columns = data_R.columns[3:]
    data_L.loc[:, shift_columns] = data_L[shift_columns].shift(
        subject_details[f"S{subject}"]["delay"][shift_key][0], fill_value=0)
    data_R.loc[:, shift_columns] = data_R[shift_columns].shift(
        subject_details[f"S{subject}"]["delay"][shift_key][1], fill_value=0)
    return data_L, data_R


def col_rearrange(data):
    return data[["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cx", "Cy", "Cz"]]


def filter_COP(data, side=None):
    """
    side: L or R to decide the forceplate limits
    """
    FP_z_lim = (0, 0.5)
    if side == 'L':
        FP_x_lim = (0, 0.5)
    elif side == "R":
        FP_x_lim = (0.5, 1)
    else:
        raise ('Side should be "L" or "R"')

    x_mid = (FP_x_lim[0]+FP_x_lim[1])/2
    z_mid = (FP_z_lim[0]+FP_z_lim[1])/2
    for i in data.index:
        # Check if foot in force plate, move COP toward foot if it's not
        if foot_in_FP(data, i, FP_x_lim):
            data.loc[i, 'Cx'] = put_cop_in_foot(current=data.loc[i, 'Cx'],
                                                p1=data.loc[i, 'X1'],
                                                p2=data.loc[i, 'X2'])

            data.loc[i, 'Cz'] = put_cop_in_foot(current=data.loc[i, 'Cz'],
                                                p1=data.loc[i, 'Z1'],
                                                p2=data.loc[i, 'Z2'])

        # Remaining is quite meanless since it won't affect calculations.
        # but it will affect COP visualizing
        # Check if COP is out side FP, of course in this case subject not in FP
        # if Cop outside in one direction, make it equal to the point behind.
        else:
            if not FP_x_lim[0] <= data.loc[i, "Cx"] <= FP_x_lim[1]:
                data.loc[i, "Cx"] = x_mid

            if not FP_z_lim[0] <= data.loc[i, "Cz"] <= FP_z_lim[1]:
                data.loc[i, "Cz"] = z_mid

    return data.drop(columns=cop_limits_columns_remove)


def foot_in_FP(data, i, FP_x_lim):
    FP_z_lim = (0, 0.5)
    if FP_x_lim[0] <= data.loc[i, 'X1'] <= FP_x_lim[1]:
        if FP_x_lim[0] <= data.loc[i, 'X2'] <= FP_x_lim[1]:
            if FP_z_lim[0] <= data.loc[i, 'Z1'] <= FP_z_lim[1]:
                if FP_z_lim[0] <= data.loc[i, 'Z2'] <= FP_z_lim[1]:
                    return True
    return False


def put_cop_in_foot(current, p1, p2, axis='X'):
    """
    current: the current COP to be checked (data.loc[i,Cz or Cx])
    p1: first position (data.loc[i,X1 or Z1])
    p2: second point (data.loc[i,X2 or Z2])
    """
    minimum = np.minimum(p1, p2)
    maximum = np.maximum(p1, p2)
    if not minimum <= current <= maximum:  # if Cx not in the foot
        # Move to the closest edge
        if np.abs(current - minimum) < np.abs(current - maximum):
            return minimum
        else:
            return maximum
    else:
        return current


def GRF_data(data_L, data_R):
    data_L = col_rearrange(data_L)
    data_R = col_rearrange(data_R)
    L_columns_names_mapper = {"Fx": "1_ground_force_vx", "Fy": "1_ground_force_vy", "Fz": "1_ground_force_vz",
                              "Cx": "1_ground_force_px", "Cy": "1_ground_force_py", "Cz": "1_ground_force_pz",
                              "Mx": "1_ground_torque_x", "My": "1_ground_torque_y", "Mz": "1_ground_torque_z"}

    R_columns_names_mapper = {"Fx": "ground_force_vx", "Fy": "ground_force_vy", "Fz": "ground_force_vz",
                              "Cx": "ground_force_px", "Cy": "ground_force_py", "Cz": "ground_force_pz",
                              "Mx": "ground_torque_x", "My": "ground_torque_y", "Mz": "ground_torque_z"}

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


input_path = f"../Data/S{subject}/{date}/Dynamics/Force_Data/"
output_path = f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data/"
files = os.listdir(input_path)
pairs = []

markers_path = f"../Data/S{subject}/{date}/Dynamics/motion_data/"
cop_limits_columns = ['time', 'X1', 'Z1', "Z2", "X2"]
cop_limits_columns_remove = ['X1', 'Z1', "Z2", "X2"]

# Get files names
trials = ["train_01", "train_02", "val", "test"]
pairs = list(map(lambda x: (
    f"S{subject}_{x}_forceplate_1.csv", f"S{subject}_{x}_forceplate_2.csv"), trials))

# Process each trial
for i, pair in enumerate(pairs):
    output_name = get_output_name(pair[0])

    # Load Left(1) and Right(2) force plates data
    for side in pair:
        markers_file = markers_path + re.sub("_forceplate_[1-2]", "", side)
        if "2.csv" in side:
            data_R = read_data(input_path+side)
        elif "1.csv" in side:
            data_L = read_data(input_path+side)

    # Match devices coordinate system
    data_L, data_R = system_match(data_L, data_R)

    # Remove the offset from the data
    data_L, data_R = remove_offset(data_L, data_R)

    # Remove the delay
    data_L, data_R = shift_data(data_L, data_R, shift_key=trials[i])

    # Add COP limits from Opti-track
    # Right side
    R_markers = pd.read_csv(markers_file, header=6,
                            usecols=[1, 68, 107, 103, 106])
    R_markers.columns = cop_limits_columns

    data_R = pd.merge(left=data_R, right=R_markers, on='time', how='inner')
    # Left side
    L_markers = pd.read_csv(markers_file, header=6,
                            usecols=[1, 14, 53, 49, 52])
    L_markers.columns = cop_limits_columns
    data_L = pd.merge(left=data_L, right=L_markers, on='time', how='inner')


#     # Rename columns and merge Left and right side
    # Filter the data
    # data_L = filter_COP(data_L, "L")
#     data_R = filter_COP(data_R,"R")
    data_L, data_R = apply_filter(data_L, data_R)
    force_data = GRF_data(data_L, data_R)

    # Save force data
    save_force_data(force_data, output_path, output_name)
