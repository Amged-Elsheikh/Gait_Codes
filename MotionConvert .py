"""
Created on Thu Jul 29 14:26:07 2021

@author: amged
"""
# Import libraries
import pandas as pd
import json
import os
import re


def get_IO_dir(subject=None, motion_type="dynamic"):
    """[summary]

    Args:
        subject (string, optional): [Subject number in the format XX. If the value is\
            None then user will be asked manually input subject's number]. Defaults to None.
        motion_type (str, optional): [Either 'static' or 'dynamic']. Defaults to "dynamic".

    Returns:
        [list]: [a list contains inputs files directories.]
        [list]: [a list contains outputs files directories]
    """
    if subject == None:
        subject = input("insert subject number: ")
    # Create motion Setting File
    date = subject_details[f"S{subject}"]["date"]
    if motion_type == "static":
        # Inputs file path
        Inputs = [f"../Data/S{subject}/{date}/Statics/S{subject}_static.csv"]
        # Outputs file path
        Outputs = [
            f"../OpenSim/S{subject}/{date}/Statics/S{subject}_static.trc"]

    elif motion_type == "dynamic":
        # Inputs folder path
        input_path = f"../Data/S{subject}/{date}/Dynamics/motion_data/"
        # Outputs folder path
        output_path = f"../OpenSim/S{subject}/{date}/Dynamics/motion_data/"
        # Get files names (trials)
        trials = ["train_01", "train_02", "val", "test"]
        Inputs = list(map(lambda x: f"S{subject}_{x}.csv", trials))
        Outputs = list(map(lambda x: f"{x}".replace('csv', 'trc'), Inputs))

        # Get inputs and outputs full directories
        Inputs = list(map(lambda file: input_path+file, Inputs))
        Outputs = list(map(lambda file: output_path+file, Outputs))
    return Inputs, Outputs


def get_markers_labels(Input):
    """
    Input: input data path. Type: string
    """
    # Getting Markers labels
    Markers_Label = pd.read_csv(Input, header=2, nrows=0).columns.values[2:]
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Do not use set because we do not want to change the order of markers
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    return unique_labels


def convert_data(Input, Output, Markers_number, subject):
    """
    Input & Output are pathes
    """
    Markers = pd.read_csv(Input, header=5, usecols=[
                          *range(0, 3*Markers_number+2)])
    Markers['Time (Seconds)'] = Markers["Frame"]/100
    # Trim trials
    trial = re.sub(".*S[0-9]*_","",Input)
    trial = re.sub("\.[a-zA-z]*","",trial)
    trials = ('train_01', 'train_02', 'val', 'test')
    if trial in trials:
        record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
        record_start = record_period['start']*100
        record_end = record_period['end']*100
        Markers = Markers.iloc[record_start:record_end+1, :]
    Markers.to_csv(Output,  sep='\t', index=False, header=False)
    num_frames = len(Markers.iloc[:, 0])
    return num_frames


def process_trc(Output, Markers_Label, num_frames):
    New_label_Coor = '\t'
    New_label_Mar = 'Frame#\tTime'
    Markers_number = len(Markers_Label)

    for i in range(0, Markers_number):
        New_label_Coor = f"{New_label_Coor}\tX{str(i+1)}\tY{str(i+1)}\tZ{str(i+1)}"
    for i in range(0, Markers_number-1):
        New_label_Mar = f"{New_label_Mar}\t{Markers_Label[i]}\t\t"
    New_label_Mar = f"{New_label_Mar}\t{Markers_Label[Markers_number-1]}\n"

    Contents = 'PathFileType\t4\t' + '(X,Y,Z)\t' + Output + '\n' \
        + 'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames' + '\n' \
        + f'100\t100\t{num_frames}\t{Markers_number}\tm\t100\t1\t{num_frames}\n'

    with open(Output, "r+") as f:
        old = f.read()  # read everything in the file
        f.seek(0)  # rewind
        f.write(Contents + New_label_Mar + New_label_Coor + '\n\n' + old)


def csv2trc(subject=None):
    if subject == None:
        subject = input("insert subject number in XX format: ")
    motion_types = ("static", "dynamic")
    flag = True
    for motion_type in motion_types:
        print(f"{motion_type} Data")
        Inputs, Outputs = get_IO_dir(subject, motion_type=motion_type)
        for Input, Output in zip(Inputs, Outputs):
            if flag:
                Markers_Label = get_markers_labels(Input)
                flag = False
            num_frames = convert_data(Input, Output, Markers_number=len(Markers_Label), subject=subject)
            process_trc(Output, Markers_Label, num_frames)


if __name__ == '__main__':
    # Load subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    csv2trc()
