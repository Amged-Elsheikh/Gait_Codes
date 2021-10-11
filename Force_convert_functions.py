import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re
import os
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

subject = "03"
setting = pd.read_csv(
    f'../settings/force_settings/S{subject}_force_settings.csv', header=None)


Weights = {"S02": 60.5, "S03": 67.8}
w = Weights[f'S{subject}']

# ## Functions

# ### System sometimes failed to send data (columns of zeros). Next function to solve this issue


def remove_system_gap(data_L, data_R):
    columns = data_L.columns[3:-1]
    data_L.loc[data_L[' Fz'] == 0, columns] = np.nan
    data_R.loc[data_R[' Fz'] == 0, columns] = np.nan
    data_L = data_L.interpolate(method="linear")
    data_R = data_R.interpolate(method="linear")
    data_L = data_L.fillna(method="bfill")
    data_R = data_R.fillna(method="bfill")
    return data_L, data_R

def system_match(data_L, data_R):

    col_names = {" Fx": "Fx", " Fy": "Fz", " Fz": "Fy",
                 " Mx": "Mx", " My": "Mz", " Mz": "My",
                 " Cx": "Cx", " Cy": "Cz", " Cz": "Cy",
                 " MocapTime": "time"}

    # System stop working someat some frames creating a gap, fill the gaps using interpolatoion
    data_L, data_R = remove_system_gap(data_L, data_R)

    # To apply rotation, change column names and change the sign for the new z-axis data.
    data_L.rename(columns=col_names, inplace=True)
    data_R.rename(columns=col_names, inplace=True)

    # Match opti-track and force Plates origins
    data_L["Cz"] = data_L["Cz"].apply(lambda x: x+0.25)
    data_R["Cz"] = data_R["Cz"].apply(lambda x: x+0.25)

    data_L["Cx"] = data_L["Cx"].apply(lambda x: x+0.25)
    data_R["Cx"] = data_R["Cx"].apply(lambda x: x+0.75)

    # Complete the rotation by getting -z
#     data_L[["Fz","Fx"]] = data_L[["Fz","Fx"]].apply(lambda x: -x)
#     data_R[["Fz","Fx"]] = data_R[["Fz","Fx"]].apply(lambda x: -x)

    return data_L, data_R


# #### Remove offset


def remove_offset(data_L, data_R, remove=True):
    if remove:
        columns = data_L.columns[3:-3]  # Choose Forces and Moments
        for col in columns:
            data_L[col] = data_L[col] - data_L.loc[5:60, col].mean()
            data_R[col] = data_R[col] - data_R.loc[5:60, col].mean()
    return data_L, data_R


# #### Apply filter

# In[22]:


f = 6  # Filter frequency
fs = 100  # Hz
low_pass = f/(fs/2)
b2, a2 = butter(N=2, Wn=low_pass, btype='lowpass')


def apply_filter(data_L, data_R):
    columns = ["Fx", "Fy", "Fz",
               "Mx", "My", "Mz",
               "Cx", "Cz"]
    for col in columns:
        #         if col=="Fy":
        #             data_L[col] = filtfilt(b2, a2, abs(data_L[col]))
        #             data_R[col] = filtfilt(b2, a2, abs(data_R[col]))
        #         else:
        data_L.loc[:, col] = filtfilt(b2, a2, data_L[col])
        data_R.loc[:, col] = filtfilt(b2, a2, data_R[col])
    # Make any force less than a 0.1*w to zero
    data_L.loc[data_L['Fy'] < 0.1*w, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']] = 0
    data_R.loc[data_R['Fy'] < 0.1*w, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']] = 0
    return data_L, data_R


# ### There is a delay in system (approxmately 10 frames)

# In[23]:


def shift_data(data_L, data_R, shift_value=10):
    shift_columns = data_R.columns[3:]
    data_L[shift_columns] = data_L[shift_columns].shift(
        shift_value, fill_value=0)
    data_R[shift_columns] = data_R[shift_columns].shift(
        shift_value, fill_value=0)
    return data_L, data_R


# ### Rename columns for OpenSim


def get_output_name(pair):
    return re.sub("_forceplate_[0-9].csv", "_grf.sto", pair)

experement_period = {"S02":{"train_01":{"start_time":500 ,"end_time":18800},
                            "train_02":{"start_time":400 ,"end_time":18700},
                            "val":     {"start_time":630 ,"end_time":20230},
                            "test":    {"start_time":300 ,"end_time": 6000}},

                     "S03":{"train_01":{"start_time":420 ,"end_time":18120},
                            "train_02":{"start_time":460 ,"end_time":19260},
                            "val":     {"start_time":600 ,"end_time":18300},
                            "test":    {"start_time":400 ,"end_time": 6900}}}

def col_rearrange(data):
    return data[["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cx", "Cy", "Cz"]]


# ### COP Filter

# In[27]:


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


# ### Process GRF


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


input_path = setting.iloc[0, 1]
output_path = setting.iloc[1, 1]
files = os.listdir(input_path)
pairs = []
def read_data(path): return pd.read_csv(path, header=31)


markers_settings = pd.read_csv(
    f'../settings/motion_settings/S{subject}_motion.csv', header=None, usecols=[0, 1])
markers_path = markers_settings.iloc[0, 1]
cop_limits_columns = ['time', 'X1', 'Z1', "Z2", "X2"]
cop_limits_columns_remove = ['X1', 'Z1', "Z2", "X2"]

# Get files names
for current in os.listdir(input_path):
    files.remove(current)
    if len(files) != -0:
        for j in files:
            if current[:16] == j[:16]:  # ensure we pick exact experement trial file
                # Make left and right pairs in a tuple
                pairs.append((current, j))

# Process each trial
for pair in pairs:
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
    data_L, data_R = shift_data(data_L, data_R, shift_value=0)

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
    data_L = filter_COP(data_L, "L")
#     data_R = filter_COP(data_R,"R")
    data_L, data_R = apply_filter(data_L, data_R)
    force_data = GRF_data(data_L, data_R)

    # Save force data
    save_force_data(force_data, output_path, output_name)
