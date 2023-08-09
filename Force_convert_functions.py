"""
This script will process the data from force plate 1 only.
"""
import json
import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt


def get_IO_dir(subject: str, trials: List[str]):
    """
    This fynction will generate inputs and outputs directories for the selected
    subject and trials.
    """
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    date = subject_details[subject]["date"]

    input_dir = Path("..", "Data", subject, date, "Dynamics", "Force_Data")
    output_dir = Path("..", "OpenSim", subject, date, "Dynamics", "Force_Data")
    files = [f"{subject}_{trial}_forceplate_1.csv" for trial in trials]
    return input_dir, output_dir, files


def trial_process(
    subject: str,
    file: str,
    trial: str,
    input_dir,
    output_dir,
    use_filter=False,
    offset_remove=False,
):
    data = pd.read_csv(input_dir / file, header=31, low_memory=False)
    # Crop the data to get the actual experement
    data = trial_period(data, subject, trial)
    # System sometimes stop sending data for few frames
    data = remove_system_gap(data)
    # Remove the delay
    data = shift_data(data, subject, trial)
    # Remove the offset from the data
    data = remove_offset(data, remove=offset_remove)
    # Apply low pass filter
    data = apply_filter(data, subject, use_filter)
    # Match devices coordinate system
    data = system_match(data)
    # Rename columns to match OpenSim default names
    force_data = GRF_data(data)
    # Save force data
    output_name = re.sub("_forceplate_[0-9].csv", "_grf.sto", file)
    save_force_data(force_data, output_dir, output_name)
    return


def trial_period(data: pd.DataFrame, subject: str, trial: str):
    """
    this function will trim the experiment to the actual experiment period and\
    create the time column
    """
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # get the actual experiment period and force plates sampling rate
    record_period = subject_details[subject]["motive_sync"][trial]
    fps = subject_details[subject]["FP_sampling_rate"]
    # Trim the experiment to the actual experiment period
    record_start = record_period["start"] * fps
    record_end = record_period["end"] * fps
    data = data.loc[data[" DeviceFrame"].between(record_start, record_end)]
    data = data.assign(time=data[" DeviceFrame"] / fps)
    return data.reset_index(drop=True)


def remove_system_gap(data: pd.DataFrame):
    """
    In some cases force plates stop recording and send zeros.it's not common
    to have exactly 0, usually the reading will be small float number.
    This function will first set these values to NaN and then perform
    linear interpolation.
    """
    # CoP is calculated based on the force and moment.
    columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
    # Zero means no data sent in almost all cases
    data.loc[data[" Fz"] == 0, columns] = np.nan
    data = data.interpolate(method="linear")
    data = data.fillna(method="bfill")
    return data


def shift_data(data: pd.DataFrame, subject, trial):
    """
    In early experements there was no external synchronization between the
    Mocap and force plates, resulting in a starting delay. This delay is
    different every time the start button is pressed to start
    new experiment/trial.

    This function will shift the data by a number of frames specified
    by the user in the experiment json file with 'delay' as a key value.
    """
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    shift_value = subject_details[subject]["delay"][trial][0]
    if shift_value != 0:
        shift_columns = [" Fx", " Fz", " Fy",
                         " Mx", " Mz", " My",
                         " Cx", " Cz", " Cy"]
        data.loc[:, shift_columns] = data[shift_columns].shift(
            shift_value, fill_value=0
        )
    return data


def remove_offset(data: pd.DataFrame, remove=True):
    """
    Force plate sensors have a small amount of offset. It can be removed bt
    finding the average offset value and substracting from the dataset if the
    user want to do so.
    """
    if remove:
        # Choose Forces and Moments
        cols = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
        data.loc[:, cols] = data.loc[:, cols] - data.loc[5:15, cols].mean()
    return data


def apply_filter(data: pd.DataFrame, subject: str, use_filter: bool):
    """
    This function is responsible for applying a Butterworth lowpass filter
    for the force and moment data when the subject is on the force plate.
    The filter low frequency will be adjusted automatically depending on
    the sample rate of the force plate.

    The function will detect when the subject is on the force plate if
    the Fz value is greater the 10% of the subject's weight.

    """
    # If the CoP is outside the force plate, put it at it's center
    data[" Cx"] = -data[" My"] / data[" Fz"]
    data[" Cy"] = data[" Mx"] / data[" Fz"]
    data.loc[abs(data[" Cx"]) > 0.25, " Cx"] = 0
    data.loc[abs(data[" Cy"]) > 0.25, " Cy"] = 0
    if not use_filter:
        return data
    # Load experiment information
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Set the trigger value and get the fps
    trigger = 0.1 * float(subject_details[subject]["weight"])
    fs = subject_details[subject]["FP_sampling_rate"]  # Hz
    # Get the periods when subject is on the force plate.
    stance_periods = grf_periods(data, trigger)
    # Create the lowpass filter based on the forceplate sampling rate
    f = fs / 20  # Filter frequency
    low_pass = f / (fs / 2)
    b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
    # select the columns to be filtered
    columns = [" Fx", " Fy", " Fz", " Mx", " My", " Mz"]
    for stance in stance_periods:
        cond = data["MocapFrame"].between(stance.start - 5, stance.stop + 5)
        data.loc[cond, columns] = filtfilt(b2, a2, data.loc[cond, columns],
                                           axis=0)
        data.loc[cond, " Cx"] = -data.loc[cond, " My"] / data.loc[cond, " Fz"]
        data.loc[cond, " Cy"] = data.loc[cond, " Mx"] / data.loc[cond, " Fz"]
    return data


def grf_periods(data: pd.DataFrame, trigger=5.0) -> List[range]:
    """
    This function will return a list of periods when the subject
    was on the force plate.
    """
    stance_periods = []
    start = None
    for i, point in enumerate(data[" Fz"]):
        # set starting point if this is the first point
        if point >= trigger and start is None:
            start = data.loc[i, "MocapFrame"]
        elif point < trigger and start is not None:
            # sometimes the subject might enter forceplate and leave
            # it immediately because (s)he enters with wrong foot. the
            # filter will raise error if the number of points is less
            # than 15 frame.
            if data.loc[i, "MocapFrame"] - start >= 20:
                stance_periods.append(slice(start, data.loc[i, "MocapFrame"]))
            start = None
    return stance_periods


def system_match(data: pd.DataFrame):
    """
    This funchion will match opti-track and force plates axes,
    so that OpenSim can correctly solve the inverse dynamic equation.
    """
    # To apply rotation, change column names.
    col_names = {
        " Fx": " Fx",
        " Fy": " Fz",
        " Fz": " Fy",
        " Mx": " Mx",
        " My": " Mz",
        " Mz": " My",
        " Cx": " Cx",
        " Cy": " Cz",
        " Cz": " Cy",
    }
    data = data.rename(columns=col_names)
    # Complete the rotation
    change_sign = [" Fx", " Fz"]
    data.loc[:, change_sign] = -data.loc[:, change_sign]
    # Match opti-track and force Plates origins
    data.loc[" Cx"] = data[" Cx"] + 0.25
    data.loc[" Cz"] = data[" Cz"] + 0.25
    return data


def GRF_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the columns name to match the names in OpenSim tutorials.
    It's possible to skip this phase, but I won't recommend it at all.
    """
    # make sure columns are well arranged for OpenSim.
    data = data[[
        "time", " Fx", " Fy", " Fz", " Mx", " My", " Mz", " Cx", " Cy", " Cz"
        ]]
    # Rename the columns
    columns_names_mapper = {
        " Fx": "1_ground_force_vx",
        " Fy": "1_ground_force_vy",
        " Fz": "1_ground_force_vz",
        " Cx": "1_ground_force_px",
        " Cy": "1_ground_force_py",
        " Cz": "1_ground_force_pz",
        " Mx": "1_ground_torque_x",
        " My": "1_ground_torque_y",
        " Mz": "1_ground_torque_z",
    }
    data = data.rename(columns=columns_names_mapper)
    return data


def save_force_data(force_data: pd.DataFrame, output_dir: Path, output_name):
    """
    This function is used to convert processed force plate data to
    OpenSim format.
    """
    output_file = output_dir / output_name
    force_data.to_csv(output_file, sep="\t", index=False)
    with open(output_file, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(
            output_name
            + "\n"
            + "version=1\n"
            + f"nRows={len(force_data)}\n"
            + f"nColumns={len(force_data.columns)}\n"
            + "inDegrees=yes\n"
            + "endheader\n"
            + old
        )
    return


def GRF_to_OpenSim(
    subject: Union[str, int],
    trials: List[str] = [
        "test",
    ],
    use_filter=False,
    offset_remove=False,
):
    """
    This function will handle finding the inputs and outputs directories,
    process the input data from csv format to sto format which will be used
    by OpenSim software later to solve the inverse dynamic equation.
    """
    try:
        subject = f"S{int(subject):02d}"
    except Exception as e:
        raise f"Exception {e}: Subject variable should be a number"
    # Get the inputs and outputs directories
    input_dir, output_dir, files = get_IO_dir(subject, trials)
    # Process each trial in different CPU
    Parallel(n_jobs=-1, verbose=10)(
        delayed(trial_process)(
            subject,
            file,
            trial,
            input_dir,
            output_dir,
            use_filter,
            offset_remove,
        )
        for file, trial in zip(files, trials)
    )
    return


if __name__ == "__main__":
    # subject = input("Please write the subject number: ")
    subject = 9
    trials = ["train_01", "train_02", "val", "test"]

    GRF_to_OpenSim(
        subject,
        trials,
        use_filter=False,
        offset_remove=False,
    )
