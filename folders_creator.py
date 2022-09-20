import json
import os


# Load subject details
with open("subject_details.json", "r") as f:
    subject_details = json.load(f)

subject = input("Please write subject number: ")
subject = f"{int(subject):02d}"
date = subject_details[f"S{subject}"]["date"]

trials = ["train_01", "train_02", "val", "test"]
setup_subs = ["GRF", "ID", "IK", "Scale"]
setup_subs = list(map(lambda x: f"{x}_setups", setup_subs))

Data = {"motions_folder": f"../Data/S{subject}/{date}/Dynamics/Motion_Data",
        "forces_folder": f"../Data/S{subject}/{date}/Dynamics/Force_Data",
        "static_folder": f"../Data/S{subject}/{date}/Statics",
        "records_folder": f"../Data/S{subject}/{date}/record_periods",
        "EMG": f"../Data/S{subject}/{date}/EMG"}

outputs = {"motions_folder": f"../OpenSim/S{subject}/{date}/Dynamics/Motion_Data",
           "forces_folder": f"../OpenSim/S{subject}/{date}/Dynamics/Force_Data",
           "static_folder": f"../OpenSim/S{subject}/{date}/Statics",
           "records_folder": f"../Outputs/S{subject}/{date}/record_periods",
           "EMG": f"../Outputs/S{subject}/{date}/EMG",
           "sEMG_dataset_folder": f"../Dataset/sEMG/S{subject}",
           "DEMG_dataset_folder": f"../Dataset/DEMG/S{subject}",
           "IK": f"../OpenSim/S{subject}/{date}/IK",
           "Model": f"../OpenSim/S{subject}/{date}/Model",
           "ID": f"../OpenSim/S{subject}/{date}/ID",
           "setups": f"../OpenSim/S{subject}/{date}/Setups"}

for folder in Data.keys():
    if not os.path.exists(Data[folder]):
        os.makedirs(Data[folder])

for folder in outputs.keys():
    if folder == "ID":
        for trial in trials:
            if not os.path.exists(f"{outputs[folder]}/{trial}"):
                os.makedirs(f"{outputs[folder]}/{trial}")
    elif folder == "setups":
        for sub in setup_subs:
            if not os.path.exists(f"{outputs[folder]}/{sub}"):
                os.makedirs(f"{outputs[folder]}/{sub}")
    else:
        if not os.path.exists(outputs[folder]):
            os.makedirs(outputs[folder])
