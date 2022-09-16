'''
This code will generate csv file(s) to find the recording periods (not 100% excat, so user need to review the points, especially for Heel Strike)
'''
import json
import re
from typing import *

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

import Force_convert_functions as grf


def get_periods(subject: Union[str, int], trilas: List[str] = ["test", ], safety_factor=5):
    try:
        subject = f"{int(subject):02d}"
    except:
        raise "Subject name should be an integer"

    # Load experiment data
    with open("subject_details.json", "r") as f:
        subject_details: Dict = json.load(f)[f"S{subject}"]
    # Get the datav and side in which experiment was conducted.
    date = subject_details["date"]
    # Useful when experiment was conducted on both sides, or when sides different from subjects. Only left side codes are made.
    sides: List[str] = subject_details["sides"]
    # Loop through trials
    for trial in trilas:
        # Start by loading Mocap data
        motive = load_heels_data(subject, trial)
        # The algorithm only uses Heel marker
        motive.columns = ['L.Heel', "R.Heel"]
        # Smooth the Mocap signal to insure better results by applying a lowpass butterworth filter
        motive.loc[:, ['L.Heel', "R.Heel"]] = lowpass_filter(
            motive.loc[:, ['L.Heel', "R.Heel"]])
        # Create two data frames, one for each side and store the in a dictionary for easy accessing
        left_periods = pd.DataFrame(columns=['left_start',
                                             'left_end', 'left_time'])
        right_periods = pd.DataFrame(columns=['right_start',
                                              'right_end', 'right_time'])
        data_dict: Dict[str, pd.DataFrame] = {
            "L": left_periods, "R": right_periods}
        # Loop in sides
        for side in sides:
            # Load GRF data
            grf_data = load_grf(subject, trial, side)
            # Get the periods when the subject was on the force plate
            stance_periods = grf.grf_periods(grf_data)
            # Loop in the stance periods
            for period in stance_periods:
                try:
                    # Using the stance period start (heel strike event) find the Toe Off event before it.
                    start = get_toe_off(
                        period.start, motive.loc[:, f"{side}.Heel"])
                    # Using the end of stance period (Toe Off event) find the following heel strike event.
                    end = get_heel_strike(
                        period.stop, motive.loc[:, f"{side}.Heel"])
                    # Take less frames when getting H.S
                    end -= safety_factor
                    # Add the new points to the df and calculate the length of the period
                    data_dict[side].loc[len(data_dict[side])] = [
                        start, end, end-start]
                except:
                    pass
        # Create the periods df
        periods = pd.concat([left_periods, right_periods])
        # Save the trial periods
        output_dir = f"../Outputs/S{subject}/{date}/record_periods/S{subject}_{trial}_record_periods.csv"
        periods.to_csv(output_dir)


def load_grf(subject: str, trial: str, side="L") -> pd.DataFrame:
    # Load experiment data
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Get the sampling rate of the force plates and experiment date
    fps = subject_details[f"S{subject}"]['FP_sampling_rate']
    date = subject_details[f"S{subject}"]["date"]
    # Generate the GRF directory
    grf_paths = f"../Data/S{subject}/{date}/Dynamics/Force_Data/S{subject}_{trial}_forceplate"
    # Make sure that forceplate 1 is for the left side, and 2 for the right side.
    if side == "L":
        grf_paths = f"{grf_paths}_1.csv"
    elif side == "R":
        grf_paths = f"{grf_paths}_2.csv"
    # Load the GRF data
    grf_data = pd.read_csv(grf_paths, header=31)
    # Remove system gaps if any (Gaps appears when not using rxternal synchorization device)
    grf_data = grf.remove_system_gap(grf_data)
    # Trim the GRF data so that we will work with actual experiment data
    grf_data = grf.trial_period(grf_data, subject, trial)
    # Downsample to 100 Hz if forceplate work on 1KHz
    if fps == 1000:
        grf_data = grf_data.loc[grf_data[' DeviceFrame'] % 10 == 0]
        grf_data.reset_index(inplace=True, drop=True)
    return grf_data


def lowpass_filter(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    f = 10  # Filter frequency in Hz
    fs = 100  # Sampling frequency in Hz
    low_pass = f/(fs/2)
    b2, a2 = butter(N=6, Wn=low_pass, btype='lowpass')
    return filtfilt(b2, a2, data, axis=0)


def get_heel_starting_index(input_path: str) -> Tuple[int]:
    """
    Find the heel markers columns id
    """
    # Getting all markers labels
    Markers_Label = pd.read_csv(input_path,
                                header=2, nrows=0
                                ).columns.values[2:]

    # print(Markers_Label[:6]) # Remove numbers in names (Uncomment this line to better understanding)
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Create a list of unique labels. Do not use set because we do not want to change the order of markers.
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    # Remove asset name if any
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    # get heel index for the df
    # multiply by 3 because in the df, each label mentioned 3 time.
    # Add 2 because first two columns are time and frame columns.
    left_heel = unique_labels.index("L.Heel")*3 + 2
    right_heel = unique_labels.index("R.Heel")*3 + 2
    # Above get for x-axis, but we care about the y-axis
    left_heel_y_column = left_heel + 1
    right_heel_y_column = right_heel + 1
    return left_heel_y_column, right_heel_y_column


def load_heels_data(subject: str, trial: str) -> pd.DataFrame:
    # Get experiment setups
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    # Get Mocap data directory
    date = subject_details[f"S{subject}"]["date"]
    motive_path = f"../Data/S{subject}/{date}/Dynamics/motion_data/S{subject}_{trial}.csv"
    # Find the columns of the heel marker (Can be done manually, but having a function can be helpful, if new markers add or some removed)
    left, right = get_heel_starting_index(motive_path)
    # Load all markers's data
    Markers = pd.read_csv(motive_path, header=5)
    # Return heels data
    return Markers.iloc[:, [left, right]]


def get_toe_off(HS_frame: int, data: pd.DataFrame, safety_factor=20) -> int:
    '''
    This algorithm will take the heel strike on forceplate frame, and do backward looping to find the toe off event before the heel strike on the force plate.
    '''
    # Start before the starting point (make it faster and can account for poor data)
    HS_frame -= safety_factor
    # Look for the point where HS at the top of a convence shape, the TO event is few frames before this point
    # (see heel's marker y-axis movement for better understanding)
    while data.iloc[HS_frame-1] >= data.iloc[HS_frame]:
        HS_frame -= 1
    return HS_frame - 5


def get_heel_strike(FP_contact_end: int, data: pd.DataFrame, safety_factor=20) -> int:
    '''
    This algorithm will take the frame after terminating the contact with the force plate, and do forward looping to find the following heel strike event frame.
    '''
    # Start looping after some extra frames (make it faster and can account for poor data)
    FP_contact_end += safety_factor
    # loop from the end of FP contact (highest point) until reaching the lowest point on the curve
    while data.iloc[FP_contact_end+1] <= data.iloc[FP_contact_end]:
        FP_contact_end += 1
    # Make sure the point is the lowest point by looking for the following 15 frames
    following_points = [data.iloc[i]
                        for i in range(FP_contact_end, FP_contact_end+15)]
    # Select the point with the minimum value as the HS event
    FP_contact_end += following_points.index(min(following_points))
    return FP_contact_end


if __name__ == '__main__':
    trilas = ["train_01", "train_02", "val", "test"]
    get_periods(subject=10, trilas=trilas)
