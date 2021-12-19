"""
Created on Thu Jul 29 14:26:07 2021

@author: amged
"""
# Import libraries
import pandas as pd
import json
import os


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


def get_markers_labels(Input, Markers_Set_Name="Amged:"):
    """
    Input: input data path. Type: string
    """
    # Getting Markers labels
    Label = pd.read_csv(Input, header=2, nrows=0).columns.values
    Markers_Label = []  # Define a list to receive markers

    for i in range(0, Markers_number):
        Temp = Label[2+3*i]
        # Need to remove the Marker set name
        Temp_1 = Temp.replace(Markers_Set_Name, "")
        Markers_Label.append(Temp_1)
    # print("Number of labels: ",len(Markers_Label))
    return Markers_Label


def convert_data(Input, Output):
    """
    Input & Output are pathes
    """
    Markers = pd.read_csv(Input, header=5, usecols=[
                          *range(0, 3*Markers_number+2)])
    if os.path.exists(Output):  # If we already have the file, it will be removed
        os.remove(Output)
    Markers.to_csv(Output,  sep='\t', index=False, header=False)
    num_frames = len(Markers.iloc[:, 0])
    return num_frames


def process_trc(Output, Markers_Label, num_frames):
    New_label_Coor = '\t'
    New_label_Mar = 'Frame#\tTime'

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
    for motion_type in motion_types:
        print(f"{motion_type} Data")
        Inputs, Outputs = get_IO_dir(subject, motion_type=motion_type)

        # if motion_type == "dynamic":
        for Input, Output in zip(Inputs, Outputs):
            Markers_Label = get_markers_labels(Input)
            num_frames = convert_data(Input, Output)
            process_trc(Output, Markers_Label, num_frames)
        # else:
        #     Markers_Label = get_markers_labels(Inputs)
        #     num_frames = convert_data(Inputs, Outputs)
        #     process_trc(Outputs, Markers_Label, num_frames)


Markers_number = 39

if __name__ == '__main__':
    # Load subject details
    with open("subject_details.json", "r") as f:
        subject_details = json.load(f)
    csv2trc()
