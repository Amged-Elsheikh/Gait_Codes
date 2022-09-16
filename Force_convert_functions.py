'''
This script will process the data from force plate 1 only. 
'''
import json
import os
import re
from typing import *

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

pd.set_option('mode.chained_assignment', None)


def GRF_to_OpenSim(subject: Union[str, int], trials: List[str] = ['test', ], 
                   use_filter: bool = False, offset_remove: bool = False) -> None:
    '''
    This function will handle finding the inputs and outputs directories, 
    process the input data from csv format to sto format which will be used
    by OpenSim software later to solve the inverse dynamic equation. 
    '''
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    # Get the inputs and outputs directories
    input_path, output_path, files = get_IO_dir(subject, trials)
    # Process each trial
    for i, file in enumerate(files):
        # Load Left force plates data
        data = pd.read_csv(input_path+file, header=31, low_memory=False)

        # Crop the data to get the actual experement
        data = trial_period(data, subject, trials[i])

        # System sometimes stop sending data for few frames
        data = remove_system_gap(data)

        # Remove the delay
        data = shift_data(data, subject, shift_key=trials[i])

        # Remove the offset from the data
        data = remove_offset(data, remove=offset_remove)

        # Apply low pass filter
        if use_filter:
            data = apply_filter(data, subject)

        # Match devices coordinate system
        data = system_match(data)

        # Rename columns to match OpenSim default names
        force_data = GRF_data(data)
        # Save force data
        output_name = re.sub("_forceplate_[0-9].csv", "_grf.sto", file)
        save_force_data(force_data, output_path, output_name)


def get_IO_dir(subject: str, trials: List[str]) -> Tuple[str, str, List[str]]:
    """
    This fynction will generate inputs and outputs directories for the selected\
    subject and trials.
    """
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)

    date = subject_details[f"S{subject}"]["date"]
    input_path = f"../Data/S{subject}/{date}/Dynamics/Force_Data/"
    output_path = f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data/"
    files = [f"S{subject}_{trial}_forceplate_1.csv" for trial in trials]
    return input_path, output_path, files

        # Match devices coordinate system
        data = system_match(data)

def trial_period(data: pd.DataFrame, subject: str, trial: str) -> pd.DataFrame:
    '''
    this function will trim the experiment to the actual experiment period and\
    create the time column
    '''
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # get the actual experiment period and force plates sampling rate
    record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
    fps = subject_details[f"S{subject}"]['FP_sampling_rate']
    # Create time column by dividing frame number by the sampling rate
    data['time'] = data[" DeviceFrame"]/fps
    # Trim the experiment to the actual experiment period
    record_start = int(record_period['start']*fps)
    record_end = int(record_period['end']*fps)
    data = data.iloc[record_start:record_end+1, :]
    # Reset the index column but keep the time column
    data.reset_index(inplace=True, drop=True)
    return data


def remove_system_gap(data: pd.DataFrame) -> pd.DataFrame:
    """
    In some cases force plates stop recording and send zeros.it's not common to have
    exactly 0, usually the reading will be small float number
    This function will first set these values to NaN and then perform linear interpolation.
    """
    # Get the main sensors' data from the dataset. CoP is calculated based on the force and moment
    # not measured by a sensor directly.
    columns = [' Fx', ' Fy', ' Fz',
               ' Mx', ' My', " Mz"]
    # Zero means no data sent in almost all cases
    data.loc[data.loc[:, ' Fz'] == 0, columns] = np.nan
    data.iloc[:, :] = data.interpolate(method="linear")
    data.iloc[:, :] = data.fillna(method="bfill")
    return data

    shift_value = subject_details[f"S{subject}"]["delay"][shift_key][0]
    if shift_value != 0:
        shift_columns = [' Fx', ' Fz', ' Fy',
                         ' Mx', ' Mz', ' My',
                         ' Cx', ' Cz', ' Cy']
        data.loc[:, shift_columns] = data[shift_columns].shift(
            shift_value, fill_value=0)
    return data

def shift_data(data: pd.DataFrame, subject, shift_key) -> pd.DataFrame:
    """
    In early experements there was no external synchronization between the Mocap
    and force plates, resulting in a starting delay. This delay is different every
    time the start button is pressed to start new experiment/trial. 

    This function will shift the data by a number of frames specified by the user
    in the experiment json file with 'delay' as a key value.
    """
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


def remove_offset(data: pd.DataFrame, remove: bool = True) -> pd.DataFrame:
    '''
    Force plate sensors have a small amount of offset. It can be removed bt
    finding the average offset value and substracting from the dataset if the 
    user want to do so.
    '''
    if remove:
        # Choose Forces and Moments
        columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
        for col in columns:
            data.loc[:, col] = data.loc[:, col] - data.loc[5:15, col].mean()
    return data


def apply_filter(data: pd.DataFrame, subject: str) -> pd.DataFrame:
    '''
    This function is responsible for applying a Butterworth lowpass filter
    for the force and moment data when the subject is on the force plate.
    The filter low frequency will be adjusted automatically depending on
    the sample rate of the force plate.

    The function will detect when the subject is on the force plate if
    the Fz value is greater the 10% of the subject's weight.

    '''
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Set the trigger value and get the fps
    trigger = 0.1 * float(subject_details[f"S{subject}"]['weight'])
    fs = subject_details[f"S{subject}"]["FP_sampling_rate"]  # Hz
    # Get the periods when subject is on the force plate.
    stance_periods = grf_periods(data, trigger)
    # Create the lowpass filter based on the forceplate sampling rate
    f = fs/20  # Filter frequency
    low_pass = f/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    # select the columns to be filtered
    columns = [" Fx", " Fy", " Fz",
               " Mx", " My", " Mz", ]
    # Recalculate the CoP
    data[' Cx'] = -data[' My']/data[' Fz']
    data[' Cy'] = data[' Mx']/data[' Fz']
    # If the CoP is outside the force plate, put it at it's center
    data[' Cx'].loc[abs(data[' Cx']) > 0.25] = 0
    data[' Cy'].loc[abs(data[' Cy']) > 0.25] = 0
    # apply the filter
    for stance in stance_periods:
        # Set the conditions for the data to be filtered
        condition = (data["MocapFrame"] >= stance.start-5) &\
            (data["MocapFrame"] <= stance.stop+5)
        # Apply the filter
        data.loc[condition, columns] = filtfilt(
            b2, a2, data.loc[condition, columns], axis=0)
    # CoP are calculated from the Force and Moment.
    # Filter CoP by recalculate it from the filtered data.
    # Note that the CoP will grow when foot outside force plate
    # since Fz is close to zero.
        data.loc[condition, " Cx"] = - \
            data.loc[condition, " My"]/data.loc[condition, " Fz"]

        data.loc[condition, " Cy"] = \
            data.loc[condition, " Mx"] / data.loc[condition, " Fz"]
    return data


def grf_periods(data: pd.DataFrame, trigger: float = 5.0) -> List[range]:
    '''
    This function will return a list of periods when the subject 
    was on the force plate.
    '''
    stance_periods = []
    start = None
    for i, point in enumerate(data[" Fz"]):
        # set starting point if this is the first point
        if point >= trigger and start == None:
            start = data.loc[i, "MocapFrame"]
        elif point < trigger and start != None:
            # sometimes the subject might enter forceplate and leave
            # it immediately because (s)he enters with wrong foot. the
            # filter will raise error if the number of points is less
            # than 15 frame.
            if data.loc[i, "MocapFrame"]-start >= 20:
                stance_periods.append(slice(start, data.loc[i, "MocapFrame"]))
            start = None
    return stance_periods


def system_match(data: pd.DataFrame) -> pd.DataFrame:
    """
    This funchion will match opti-track and force plates axes,
    so that OpenSim can correctly solve the inverse dynamic equation.
    """
    # To apply rotation, change column names.
    col_names = {" Fx": " Fx", " Fy": " Fz", " Fz": " Fy",
                 " Mx": " Mx", " My": " Mz", " Mz": " My",
                 " Cx": " Cx", " Cy": " Cz", " Cz": " Cy", }
    data.rename(columns=col_names, inplace=True)
    # Complete the rotation
    change_sign = [" Fx", " Fz"]
    data.loc[:, change_sign] = -data.loc[:, change_sign]
    # Match opti-track and force Plates origins
    data.loc[:, " Cx"] = data[" Cx"] + 0.25
    data.loc[:, " Cz"] = data[" Cz"] + 0.25
    return data


def GRF_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the columns name to match the names in OpenSim tutorials.
    It's possible to skip this phase, but I won't recommend it at all.
    """
    # make sure columns are well arranged to save time when working with OpenSim.
    data = data[["time", " Fx", " Fy", " Fz",
                 " Mx", " My", " Mz",
                 " Cx", " Cy", " Cz"]]
    # Rename the columns
    columns_names_mapper = {" Fx": "1_ground_force_vx",
                              " Fy": "1_ground_force_vy",
                              " Fz": "1_ground_force_vz",
                              " Cx": "1_ground_force_px",
                              " Cy": "1_ground_force_py",
                              " Cz": "1_ground_force_pz",
                              " Mx": "1_ground_torque_x",
                              " My": "1_ground_torque_y",
                              " Mz": "1_ground_torque_z"}
    data.rename(columns=columns_names_mapper, inplace=True)
    return data


def save_force_data(force_data: pd.DataFrame, output_path: str, output_name: str) -> None:
    '''
    This function is used to convert processed force plate data (Pandas/CSV) to OpenSim 
    format.
    '''
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
    subject = input('Please write the subject number: ')
    trials = ["train_01", "train_02", "val", "test"]

    GRF_to_OpenSim(subject=subject, trials=trials,
                   use_filter=False, offset_remove=False,)
