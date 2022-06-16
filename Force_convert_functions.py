import json
import os
import re
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)


def get_IO_dir(subject, trials):
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        
    date = subject_details[f"S{subject}"]["date"]
    input_path = f"../Data/S{subject}/{date}/Dynamics/Force_Data/"
    output_path = f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data/"
    files = [f"S{subject}_{trial}_forceplate_1.csv" for trial in trials]
    return input_path, output_path, files


def remove_system_gap(data):
    """
    In some cases force plates stop recording and send only zeros. This function will\\
        first set these values for NaN and then interpolate missing values.
    """
    columns = [' Fx', ' Fy', ' Fz',
               ' Mx', ' My', " Mz"]

    data.loc[data.loc[:, ' Fz'] == 0, columns] = np.nan
    data.iloc[:, :] = data.interpolate(method="linear")
    data.iloc[:, :] = data.fillna(method="bfill")
    return data


def trial_period(data, subject, trial):
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
    fps = subject_details[f"S{subject}"]['FP_sampling_rate']
    data['time'] = data[" DeviceFrame"]/fps
    record_start = int(record_period['start']*fps)
    record_end = int(record_period['end']*fps)
    data = data.iloc[record_start:record_end+1, :]
    data.reset_index(inplace=True, drop=True)
    return data


def remove_offset(data, remove=True):
    if remove:
        # Choose Forces and Moments
        columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
        for col in columns:
            data.loc[:, col] = data.loc[:, col] - data.loc[5:15, col].mean()
    return data


def grf_periods(data, trigger=5):
    stance_periods = []
    start = None
    for i, point in enumerate(data[" Fz"]):
        # set starting point if this is the first point
        if point >= trigger and start == None:
            start = data.loc[i, "MocapFrame"]
        elif point < trigger and start != None:
            # sometimes the subject might enter forceplate and leave it immediately because he enter with wrong foot. the filter will raise error if the number of points is less than 15
            if data.loc[i, "MocapFrame"]-start >= 20:
                stance_periods.append(slice(start, data.loc[i, "MocapFrame"]))
            start = None
    return stance_periods


def apply_filter(data, subject, trigger=5):
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    
    stance_periods = grf_periods(data, trigger)
    f = 10  # Filter frequency
    fs = subject_details[f"S{subject}"]["FP_sampling_rate"]  # Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    columns = [" Fx", " Fy", " Fz",
               " Mx", " My", " Mz", ]

    data[' Cx'] = -data[' My']/data[' Fz']
    data[' Cy'] = data[' Mx']/data[' Fz']
    data[' Cx'].loc[abs(data[' Cx'])>0.25] = 0
    data[' Cy'].loc[abs(data[' Cy'])>0.25] = 0
    # apply filter
    for stance in stance_periods:
        condition = (data["MocapFrame"] >= stance.start-5) & (
            data["MocapFrame"] <= stance.stop+5)
        data.loc[condition, columns] = filtfilt(
            b2, a2, data.loc[condition, columns], axis=0)
    ##     CoP are calculated from the Force and Moment. Filter CoP by recalculate it from the filtered data. Note that the CoP will grow when foot outside force plate.
        data.loc[condition, " Cx"] = -data.loc[condition, " My"]/data.loc[condition, " Fz"]
        data.loc[condition, " Cy"] = data.loc[condition," Mx"] / data.loc[condition, " Fz"]
    return data


def system_match(data):
    """
    Match opti-track and force plates axes.
    Remove system gaps if any.
    """
    # To apply rotation, change column names.
    col_names = {" Fx": " Fx", " Fy": " Fz", " Fz": " Fy",
                 " Mx": " Mx", " My": " Mz", " Mz": " My",
                 " Cx": " Cx", " Cy": " Cz", " Cz": " Cy", }
    data.rename(columns=col_names, inplace=True)
    # Match opti-track and force Plates origins
    data.loc[:, " Cx"] = data[" Cx"] + 0.25
    data.loc[:, " Cz"] = data[" Cz"] + 0.25
    # Complete the rotation
    change_sign = [" Fx", " Fz"]
    data.loc[:, change_sign] = -data.loc[:, change_sign]
    return data


# There is a delay in system (various delay may change )
def shift_data(data, shift_key):
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
        
    shift_value = subject_details[f"S{subject}"]["delay"][shift_key][0]
    if shift_value != 0:
        shift_columns = [' Fx', ' Fz', ' Fy',
                         ' Mx', ' Mz', ' My',
                         ' Cx', ' Cz', ' Cy']
        data.loc[:, shift_columns] = data[shift_columns].shift(
            shift_value, fill_value=0)
    return data


def col_rearrange(data):
    return data[["time", " Fx", " Fy", " Fz", " Mx", " My", " Mz", " Cx", " Cy", " Cz"]]


def GRF_data(data):
    # make sure columns are weell arranged
    data = col_rearrange(data)
    L_columns_names_mapper = {" Fx": "1_ground_force_vx",
                              " Fy": "1_ground_force_vy",
                              " Fz": "1_ground_force_vz",
                              " Cx": "1_ground_force_px",
                              " Cy": "1_ground_force_py",
                              " Cz": "1_ground_force_pz",
                              " Mx": "1_ground_torque_x",
                              " My": "1_ground_torque_y",
                              " Mz": "1_ground_torque_z"}
    data.rename(columns=L_columns_names_mapper, inplace=True)
    return data


def save_force_data(force_data, output_path, output_name):
    output_file = output_path + output_name
    if os.path.exists(output_file):
        os.remove(output_file)
    force_data.to_csv(output_file,  sep='\t', index=False)
    nRows = len(force_data)  # end_time - start_time + 1
    nColumns = len(force_data.columns)

    with open(output_file, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(output_name+'\n' + 'version=1\n' +
                f'nRows={nRows}\n' + f'nColumns={nColumns}\n' + 'inDegrees=yes\n' + 'endheader\n' + old)


if __name__ == '__main__':
    # Get files names
    trials = ["train_01", "train_02", "val", "test"]
    subject = input(f"insert subject number in XX format: ")
    # for subject in ["10",'11']:

    input_path, output_path, files = get_IO_dir(subject, trials)

    # Process each trial
    for i, file in enumerate(files):
        # Load Left force plates data
        data = pd.read_csv(input_path+file, header=31, low_memory=False)

        # prepare time
        data = trial_period(data, subject, trials[i])

        # System sometimes stop sending data for few frames
        data = remove_system_gap(data)

        # Remove the delay
        data = shift_data(data, shift_key=trials[i])

        # Remove the offset from the data
        # data = remove_offset(data)

        # Apply low pass filter
        # data = apply_filter(data, subject)

        # Match devices coordinate system
        data = system_match(data)

        # Rename columns to match OpenSim default names
        force_data = GRF_data(data)
        # Save force data
        output_name = re.sub("_forceplate_[0-9].csv", "_grf.sto", file)
        save_force_data(force_data, output_path, output_name)
