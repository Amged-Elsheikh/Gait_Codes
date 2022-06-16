# Import libraries
import pandas as pd
import json
import re


def get_IO_dir(subject=None, motion_type="dynamic"):

    # If subject number not specified, user should write it manually
    if subject == None:
        subject = input("insert subject number: ")
    # Create motion Setting File
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
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
    # Getting markers_trajectories labels
    Markers_Label = pd.read_csv(Input, header=2, nrows=0).columns.values[2:]
    Markers_Label = list(
        map(lambda x: re.sub('\.[0-9]$', "", x), Markers_Label))
    # Do not use set because we do not want to change the order of markers_trajectories
    unique_labels = []
    for label in Markers_Label:
        if label not in unique_labels:
            unique_labels.append(label)
    unique_labels = list(map(lambda x: re.sub('.+:', "", x), unique_labels))
    return unique_labels


def load_trajectories(subject, Input):
    """
    Input & Output are pathes
    """
    markers_trajectories = pd.read_csv(Input, header=5)
    markers_trajectories['Time (Seconds)'] = markers_trajectories["Frame"]/100
    # Get trial name
    trial = re.sub(".*S[0-9]*_", "", Input)
    trial = re.sub("\.[a-zA-z]*", "", trial)
    trials = ('train_01', 'train_02', 'val', 'test')

    if trial in trials:
        with open("subject_details.json", "r") as f:
            subject_details = json.load(f)

        record_period = subject_details[f"S{subject}"]["motive_sync"][trial]
        record_start = int(record_period['start'] * 100)
        record_end = int(record_period['end'] * 100)
        markers_trajectories = markers_trajectories.iloc[record_start:record_end + 1, :]

    return markers_trajectories


def process_trc(markers_trajectories, Output, Markers_Label):
    New_label_Coor = '\t'
    New_label_Mar = 'Frame#\tTime'
    Markers_number = len(Markers_Label)
    num_frames = len(markers_trajectories)

    markers_trajectories.to_csv(Output,  sep='\t', index=False, header=False)
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


def csv2trc(subject=None, motion_types=None):
    if subject == None:
        subject = input("insert subject number in XX format: ")

    if motion_types == None:
        motion_types = ("static", "dynamic")
    elif type(motion_types) == str:
        motion_types = [motion_types, ]

    flag = True
    for motion_type in motion_types:
        print(f"{motion_type} Data")
        # Get inputes and outputs directories
        Inputs, Outputs = get_IO_dir(subject, motion_type=motion_type)
        # Loop in trials
        for Input, Output in zip(Inputs, Outputs):
            # Get experement labels once
            if flag:
                Markers_Label = get_markers_labels(Input)
                flag = False

            markers_trajectories = load_trajectories(subject, Input)
            process_trc(markers_trajectories, Output, Markers_Label)


if __name__ == '__main__':
    # Load subject details
    csv2trc(subject=None, motion_types=None)
