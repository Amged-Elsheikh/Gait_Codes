#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[25]:


# import shutup
# shutup.please()
import pandas as pd
import numpy as np
import os
# import plotly.express as px
# import matplotlib.pyplot as plt
# import plotly.io as pio
# pio.renderers.default = "browser"

Weights ={"S02":60.5}
# ## Define functions To extract gait events

# ### Load data

# In[36]:


def get_files(settings):
    """
    return motion and GRF and outputs paths
    """
    motion_folder = settings.iloc[0,1]
    motion_files = os.listdir(motion_folder)
    motion_paths = list(map(lambda x: motion_folder + x, motion_files))
    
    grf_folder = settings.iloc[1,1]
    grf_files = os.listdir(grf_folder)
    grf_paths = list(map(lambda x: grf_folder + x, grf_files))
    grf_pairs_path = []
    for i in range(0, len(grf_paths), 2):
        grf_pairs_path.append((grf_paths[i],grf_paths[i+1]))
        
    outputs_path = settings.iloc[2,1]
    output_names = list(map(lambda x: x.replace('.csv','_events.csv'), motion_files))
    output_names = list(map(lambda x: outputs_path + x, output_names))
    
    return motion_paths, grf_pairs_path, output_names
    
def load_motion(motion_path):
    """
    This version will use Heel marker only
    """
    # Load Heels Y-axis position data
    markers_data = pd.read_csv(motion_path, skiprows=6, usecols=[1, 15, 69])

    # Rename the columns
    markers_data.columns = ["time",'L.Heel', 'R.Heel'] 

    # Get the gradient for each column
    markers_data = get_gradiant(markers_data, ["L.Heel", "R.Heel"])
    return markers_data


def get_gradiant(data, col_names):
    if type(col_names)==list:
        for col in col_names:
            data[f"{col}_grad"] = pd.DataFrame(np.gradient(data[col]))
    else:
        data[f"{col_names}_grad"] = pd.DataFrame(np.gradient(data[col_names]))
    return data


def load_grf(grf_pairs):
    """
    Input is a tuple (left_grf, right_grf)
    """
    L_grf_path = grf_pairs[0]
    R_grf_path = grf_pairs[1]
    
    L_grf_data = pd.read_csv(L_grf_path, header=31, usecols=[1,5])
    L_grf_data.columns = ['time','Fz']
    
    R_grf_data = pd.read_csv(R_grf_path, header=31, usecols=[1,5])
    R_grf_data.columns = ['time','Fz']
    return L_grf_data, R_grf_data


# ### Fining events

# In[37]:


def find_HS(data, sides, HS_threshold=0.05):
    """
    the function find almost all possible candidate locations for heel strike, by applying a set of conditions:
    1. Sign change in the gradient.
    2. Elevation  threshold.
    3. It's not possible to have two HS in half second.
    
    side is a string of either L or R or both as a tuple
    
    The column will be filled by the position value to be able to plot later
    """
    for side in sides:
        data[f'{side}.HS'] = np.nan
        for i in data.index:
            # First condition is when derivative sign changed from negative to posative and remain posative
            if data.loc[i, f"{side}.Heel_grad"] < 0 and all(data.loc[i+1:i+4, f"{side}.Heel_grad"] > 0):

                # Second condition, make sure all points below specific level
                if data.loc[i, f"{side}.Heel"]<=HS_threshold:

                    # Third condition it's not possible to have 2 HS in half second
                    if data.loc[i-30:i, f"{side}.HS"].isnull().all():
                        data.loc[i, f"{side}.HS"] = data.loc[i, f"{side}.Heel"]

def find_TO(data, sides, TO_threshold=0.05):
    """
    the function find almost all possible candidate locations for Toe Off, by applying a set of conditions:
    1. Elevation  threshold.
    2. derivative sign change.
    
    side is a string of either L or R or both as a tuple
    
    The column will be filled by the position value to be able to plot later
    """
    for side in sides:
        data[f"{side}.TO"] = np.nan
        for i in data.index:
            # First condition, make sure all points above specific level
            if data.loc[i, f"{side}.Heel"] >= TO_threshold:
                # Second condition is when derivative sign changed from posative to negative and remain negative
                if data.loc[i, f"{side}.Heel_grad"] > 0 and all(data.loc[i+1:i+8, f"{side}.Heel_grad"] < 0):
                    data.loc[i-3, f"{side}.TO"] = data.loc[i, f"{side}.Heel"]


# In[38]:


def find_event(data, sides=["L","R"], HS_threshold=0.05, TO_threshold=0.06):
    """
    data file should contains left and right heel markers y-axis data and their gradient
    """
    find_HS(data, sides=sides, HS_threshold=HS_threshold)
    find_TO(data, sides=sides, TO_threshold=TO_threshold)
    for side in sides:
        data[f"{side}.event"] = np.nan
        current_event="swing"
        
        for i in data.index:
            # Case one if no heel strike nor toe off assign swing or stance according to last event appeared (TO or HS)
            if (np.isnan(data.loc[i, f"{side}.HS"])) & (np.isnan(data.loc[i, f"{side}.TO"])):
                data.loc[i, f"{side}.event"] = current_event

            # If HS event assign HS to data["event"] and update current_event to stance, note that before HS their must be swing event
            elif (not np.isnan(data.loc[i, f"{side}.HS"])) & (current_event=="swing"):
                data.loc[i, f"{side}.event"] = "HS"
                current_event = "stance"

            # If TO event assign 'TO' to data["event"] and update current_event to stance
            elif not np.isnan(data.loc[i, f"{side}.TO"]) & (current_event=="stance"):
                data.loc[i, f"{side}.event"] = "TO"
                current_event = "swing"
            # If no condition met, assign to the previous event
            else:
                data.loc[i, f"{side}.event"] = data.loc[i-1, f"{side}.event"]

    return data
# ### Finding FP intervals

# In[ ]:


def on_FP(force_data, side, Subject_weight, weight_threshold=0.1):
    """
    Create a column that tells eaither subject is stepping on Force Plate (True) or no (False)
    """
    force_data[f'{side}_On.FP'] = False
    force_data[f'{side}_On.FP'].loc[force_data["Fz"] >= Subject_weight*weight_threshold]= True
    return force_data


def data_selector(markers_data, grf_data, side):
    """
    Select two nearby T.O and if there are any True in On.FP column, set all values to True in the new column
    Do the same with heels
    
    For setting True, you can add some saftey range for examble delay 10 frames at the start and stop 10 frames earlier
    """
    
    data = pd.merge(left=markers_data[['time',f'{side}.event']], right=grf_data[['time',f'{side}_On.FP']], on='time', how='inner')
    # Make sure merge was done for all data points
    assert len(markers_data) == len(data)
    
    ## get the time of TO and HS, multiply by 100 and convert to integer to get the index of the event
    to_index = np.array(data['time'].loc[data[f'{side}.event']=='TO'].values*100, dtype='int32')
    hs_index = np.array(data['time'].loc[data[f'{side}.event']=='HS'].values*100, dtype='int32')

    data[f'{side}_side_select'] = data[f'{side}_On.FP']
    
    for i in range(len(to_index)-1):
        start = to_index[i] # start 10 frames after TO
        end   = to_index[i+1] # End 10 frames earlier
        if np.sum(data.loc[start:end, f'{side}_On.FP'])>20:
            data.loc[start:end, f'{side}_side_select'] = True

    for i in range(len(hs_index)-1):
        start = to_index[i] #start 10 frames before HS (while swinging)
        end = to_index[i+1] # End 10 frames earlier
        if np.sum(data.loc[start:end, f'{side}_On.FP'])>20:
            data.loc[start:end, f'{side}_side_select'] = True
    return data[['time', f'{side}_side_select']]


# ### process the data

# In[33]:

def true_check(data, check_length=5):
    L = check_length
    for side in ["L","R"]:
        for i in data[L:-L].index:
            if data.loc[i, f'{side}_side_select']==True:
                if all(data.loc[i:i+L, f'{side}_side_select']) or all(data.loc[i-L:i, f'{side}_side_select']):
                    pass
                else:
                    data.loc[i-L:i+L, f'{side}_side_select']=False
    return data

def process_event_file(subject=None, HS_threshold=0.05, TO_threshold=0.055, weight_threshold=0.1):
    if subject == None:
        subject = input("Input subject number in XX format: ")
    # Load settings file
    settings = pd.read_csv(f'../settings/Events_settings/S{subject}_events.csv', header = None, usecols = [0,1])
    # get subject weight
    Subject_weight = Weights[f"S{subject}"]
    # get all files paths from the settings file
    motion_file, grf_pairs_path, output_names = get_files(settings)
    # start working on each trial
    for motion_path, grf_pairs, output_path in\
                                zip(motion_file, grf_pairs_path, output_names):
        # Load data
        markers_data = load_motion(motion_path)
        L_grf_data, R_grf_data = load_grf(grf_pairs)
        # Make sure we are loading same trial data
        assert len(markers_data) == len(L_grf_data) == len(R_grf_data)
        ## Use Markers data to find events
        markers_data = find_event(markers_data, sides=["L","R"],
                                  HS_threshold=HS_threshold, TO_threshold=TO_threshold)
        ## Use force data and subject weight to find if subject on Force plate or no
        L_grf_data = on_FP(L_grf_data, 'L', Subject_weight=Subject_weight, 
                           weight_threshold=weight_threshold)
        
        R_grf_data = on_FP(R_grf_data, 'R', Subject_weight=Subject_weight, 
                           weight_threshold=weight_threshold)
        
        # Select which data to be selected for each side
        L_data = data_selector(markers_data, L_grf_data, side="L")
        R_data = data_selector(markers_data, R_grf_data, side="R")

        data = pd.merge(left=L_data, right=R_data, on='time', how='inner')
        assert len(R_data)==len(data)
        data = true_check(data)
        data.to_csv(output_path, index=False)

# In[39]:


process_event_file('02')


# ### Plots

# In[35]:


# import plotly.graph_objects as go

# fig1 = px.line(data[:],x='time', y="R.Heel")
# fig2 = go.Figure(data=go.Scatter(x=data["time"], y=data["R.HS"], mode='markers', marker_color='rgba(255, 0, 0, 1)', text="Heel Strike", name="H.S"))
# fig3 = go.Figure(data=go.Scatter(x=data["time"], y=data["R.TO"], mode='markers', marker_color='rgba(0, 255, 0, 1)', text="Toe Off", name="T.O"))
# ## Uncomment line for the plot
# # go.Figure(data=(fig1.data + fig2.data + fig3.data))


# In[39]:


# Left_side_plot = px.scatter(data[:],x='time', y="L.Heel", color="L.event")
# Left_side_plot


# In[40]:


# Right_side_plot = px.scatter(data, x='time', y="R.Heel", color="R.event")
# Right_side_plot

