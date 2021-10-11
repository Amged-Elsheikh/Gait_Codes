"""
Created on Thu Jul 29 14:26:07 2021

@author: amged
"""

import pandas as pd
import os


def get_IO_dir(subject=None, motion_type="dynamic"):
    # Load Motion Setting File
    setting = pd.read_csv(
        f'../settings/motion_settings/S{subject}_motion.csv', header=None, usecols=[0, 1])
    if motion_type == "static":
        input_path = setting.iloc[2, 1]  # Inputs folder
        output_path = setting.iloc[3, 1]  # Outputs folder
    elif motion_type == "dynamic":
        input_path = setting.iloc[0, 1]  # Inputs folder
        output_path = setting.iloc[1, 1]  # Outputs folder

    print("The print will only shows files names, not full directory")
    Inputs = os.listdir(input_path)
    print("Input files: ", Inputs)
    Outputs = list(map(lambda x: f"{x}".replace('csv', 'trc'), Inputs))
    print("Output files: ", Outputs)

    # Get inputs and outputs full directories
    Inputs = list(map(lambda file: input_path+file, Inputs))
    Outputs = list(map(lambda file: output_path+file, Outputs))
    return Inputs, Outputs


def get_markers_labels(Input, Markers_Set_Name="Amged:"):

    # Getting Markers labels
    Label = pd.read_csv(Input, header=2, nrows=0).columns.values
    Markers_Label = []  # Define a list to receive markers
    Markers_Set_Name = Markers_Set_Name

    for i in range(0, Markers_number):
        Temp = Label[2+3*i]
        # Need to remove the Marker set name
        Temp_1 = Temp.replace(Markers_Set_Name, "")
        Markers_Label.append(Temp_1)
#     print("Number of labels: ",len(Markers_Label))
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


def process_trc(Output, Markers_Label, num_frames, Markers_number=39):
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

    motion_types = ("dynamic", "static")
    for motion_type in motion_types:
        print(f"{motion_type} Data")
        Inputs, Outputs = get_IO_dir(subject, motion_type=motion_type)
        for Input, Output in zip(Inputs, Outputs):
            Markers_Label = get_markers_labels(Input)
            num_frames = convert_data(Input, Output)
            process_trc(Output, Markers_Label, num_frames)


Markers_number = 39
csv2trc("03")
