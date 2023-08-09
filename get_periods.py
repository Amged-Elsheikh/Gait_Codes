"""
This code will generate csv file(s) to find the recording periods
(not 100% accurate, so user need to review the points,
especially for Heel Strike)
"""
import json
from typing import Union, List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import Force_convert_functions as grf


def get_periods(
    subject: Union[str, int],
    trilas: List[str] = [
        "test",
    ],
    safety_factor=5,
):
    try:
        subject = f"S{int(subject):02d}"
    except Exception as e:
        raise f"Exeption {e}: Subject name should be an integer"

    # Load experiment data
    with open("subject_details.json", "r") as f:
        subject_details: Dict = json.load(f)[subject]
    # Get the date and side in which experiment was conducted.
    date = subject_details["date"]
    sides = subject_details["sides"]
    # Loop through trials
    for trial in trilas:
        # Start by loading Mocap data
        motive = load_heels_data(subject, trial)
        # Smooth the Mocap signal to insure better results by applying
        # a lowpass butterworth filter
        motive.loc[:] = lowpass_filter(motive)
        # Create two data frames, one for each side and store the in a
        # dictionary for easy accessing
        left_periods = pd.DataFrame(columns=["left_start", "left_end", "left_time"])
        right_periods = pd.DataFrame(columns=["right_start", "right_end", "right_time"])
        data_dict: Dict[str, pd.DataFrame] = {"L": left_periods, "R": right_periods}
        # Loop in sides
        for side in sides:
            # Load GRF data
            grf_data = load_grf(subject, trial, side)
            # Get the periods when the subject was on the force plate
            stance_periods = grf.grf_periods(grf_data)
            # Loop in the stance periods
            for period in stance_periods:
                try:
                    # Using the stance period start (heel strike event)
                    # find the Toe Off event before it.
                    start = get_toe_off(period.start, motive.loc[:, f"{side}.Heel"])
                    # Using the end of stance period (Toe Off event)
                    # find the following heel strike event.
                    end = get_heel_strike(period.stop, motive.loc[:, f"{side}.Heel"])
                    # Take less frames when getting H.S
                    end -= safety_factor
                    # Add the new points to the df and calculate the
                    # length of the period
                    data_dict[side].loc[len(data_dict[side])] = [
                        start,
                        end,
                        end - start,
                    ]
                except Exception:
                    pass
        # Create the periods df
        periods = pd.concat([left_periods, right_periods])
        # # Save the trial periods
        output_dir = f"../Outputs/S{subject}/{date}/record_periods/S{subject}_{trial}_record_periods.csv"
        periods.to_csv(output_dir)
        return


def load_grf(subject: str, trial: str, side="L"):
    # Load experiment data
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Get the sampling rate of the force plates and experiment date
    fps = subject_details[subject]["FP_sampling_rate"]
    date = subject_details[subject]["date"]
    # Generate the GRF directory
    n = 1 if side == "L" else 2
    grf_paths = Path(
        "..",
        "Data",
        subject,
        date,
        "Dynamics",
        "Force_Data",
        f"{subject}_{trial}_forceplate_{n.csv}",
    )
    # Load the GRF data
    grf_data = pd.read_csv(grf_paths, header=31)
    # Remove system gaps if any (Gaps appears when not using external
    # synchorization device)
    grf_data = grf.remove_system_gap(grf_data)
    # Trim the GRF data so that we will work with actual experiment data
    grf_data = grf.trial_period(grf_data, subject, trial)
    # Downsample to 100 Hz if forceplate work on 1KHz
    if fps == 1000:
        grf_data = grf_data.loc[grf_data[" DeviceFrame"] % 10 == 0]
        grf_data = grf_data.reset_index(drop=True)
    return grf_data


def lowpass_filter(data: Union[pd.DataFrame, np.ndarray]):
    f = 10  # Filter frequency in Hz
    fs = 100  # Sampling frequency in Hz
    low_pass = f / (fs / 2)
    b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
    return filtfilt(b2, a2, data, axis=0)


def get_heel_starting_index(input_path: str):
    """
    Find the heel markers columns id
    """
    # Getting all markers labels
    markers = pd.read_csv(input_path, header=2, nrows=0).columns[2:]
    unique_labels = [markers[x] for x in range(0, len(markers), 3)]
    unique_labels = [x.split(":")[1] for x in unique_labels]
    left_heel = unique_labels.index("L.Heel") * 3
    right_heel = unique_labels.index("R.Heel") * 3
    left_heel_y = left_heel + 1
    right_heel_y = right_heel + 1
    return left_heel_y, right_heel_y


def load_heels_data(subject: str, trial: str):
    # Get experiment setups
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Get Mocap data directory
    date = subject_details[subject]["date"]
    motive_path = Path(f"../Data/{subject}/{date}/Dynamics/motion_data/{subject}_{trial}.csv")
    # Find the columns of the heel marker (Can be done manually, but having
    # a function can be helpful, if new markers add or some removed)
    left, right = get_heel_starting_index(motive_path)
    # Load all markers's data
    heel_df = pd.read_csv(motive_path, usecols=[left + 2, right + 2], header=5)
    heel_df.columns = ["L.Heel", "R.Heel"]
    # Return heels data
    return heel_df


def get_toe_off(HS_frame: int, data: pd.DataFrame, safety_factor=20):
    """
    This algorithm will take the heel strike on forceplate frame, and do
    backward looping to find the toe off event before the heel strike on
    the force plate.
    """
    # Start before the starting point
    HS_frame -= safety_factor
    # Look for the point where HS at the top of a convence shape, the TO event
    # is few frames before this point
    # (see heel's marker y-axis movement for better understanding)
    while data.iloc[HS_frame - 1] >= data.iloc[HS_frame]:
        HS_frame -= 1
    return HS_frame - 5


def get_heel_strike(FP_contact_end: int, data: pd.DataFrame, safety_factor=20):
    """
    This algorithm will take the frame after terminating the contact with the
    force plate, and do forward looping to find the following heel strike
    event frame.
    """
    # Start looping after some extra frames
    # (make it faster and can account for poor data)
    FP_contact_end += safety_factor
    # loop from the end of FP contact
    # (highest point) until reaching the lowest point on the curve
    while data.iloc[FP_contact_end + 1] <= data.iloc[FP_contact_end]:
        FP_contact_end += 1
    # Make sure the point is the lowest point by looking at the next 15 frames
    following_points = [
        data.iloc[i] for i in range(FP_contact_end, FP_contact_end + 15)
    ]
    # Select the point with the minimum value as the HS event
    FP_contact_end += following_points.index(min(following_points))
    return FP_contact_end


if __name__ == "__main__":
    trilas = ["train_01", "train_02", "val", "test"]
    get_periods(subject=10, trilas=trilas)
