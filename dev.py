import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn import tree

from utils import dater
from utils import plotter
from utils import explainer

data_type = 'anomalies' # 'anomalies' or 'background
resolution = 'high' # 'high', 'med', 'low'

if resolution == 'high':
    window_size = 32
elif resolution == 'med':
    window_size = 16
elif resolution == 'low':
    window_size = 8

# Load a model. I am using the last model in this case -- 9.
iteration = 7
filename = f'models/rf_model_{resolution}_{iteration}.sav'
model = pickle.load(open(filename, 'rb'))

# Read the data: anomalies or background
if data_type == 'anomalies':
    
    # Load the anomalies data
    file_anomalies = open(f'pickels/anomaly_data_test.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Get the data corresponding to the resolution of the model and rename it to X
    if resolution == 'high':
        X = anomalies_windows[0][0]
    elif resolution == 'med':
        X = anomalies_windows[0][1]
    elif resolution == 'low':
        X = anomalies_windows[0][2]
    
    lengths = anomalies_windows[-1]
    number_windows = [i - window_size + 1 for i in lengths]

elif data_type == 'background':
    
    # Load the background data
    file_background = open(f'pickels/background_data_test.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close

    # Get the data corresponding to the resolution of the model and rename it to X
    if resolution == 'high':
        X = background_windows[0][0]
    elif resolution == 'med':
        X = background_windows[0][1]
    elif resolution == 'low':
        X = background_windows[0][2]
    
    lengths = background_windows[-1]
    number_windows = [i - window_size + 1 for i in lengths]

# Find the start and end window index for each resolution and save it in a 2D list: [[[high],[med],[low]],
    #                                                                               [[high],[med],[low]], ... ,]

starts_ends = []
for event_number in range(len(number_windows)):

    event_start_high = sum(number_windows[:event_number]) + event_number
    event_end_high = sum(number_windows[:event_number + 1]) + 1 + event_number

    starts_ends.append([event_start_high, event_end_high])

# 1. Given an event_number, get the indices of the windows that correspond to that event at the different resolutions
# The objects event_start and event_end give the indices of the high resolution windows.
# The first anomaly has 14 windows (counting 0, Python indexing, and not counting 14)
# Therefore, in the first event, the number of med and low resolution windows is:
event_start_med = 0
event_end_med = (number_windows[event_number] + 1) * 17
event_start_low = 0
event_end_low = (number_windows[event_number] + 1) * 25

# However, this works for the first event. In the subsiquent cases I have to take into account the number of windows
# that came before.

# 2. Once I have the indices, I can select the windows I want to plot and explain

# # Plot a given anomaly (tested and verified)
# event_number = 0 # anomaly or background event number

# event_start_high = starts_ends[event_number][0]
# event_end_high = starts_ends[event_number][1]

# event_data = X[event_start_high]
# for i in range(event_start_high + 1, event_end_high):
    
#     # Get the last row of the anomaly
#     last_row = X[i][-6:]
    
#     # Add the last row to anomaly_data
#     event_data = np.concatenate((event_data, last_row), axis=0)

# Tested an verified (plot a given anomaly)

# plotter(data=event_data, num_variables=6, windowed=False)

# X = X[:20] # Subset the data to the first 20 anomalies just for testing purposes
# explainer(X, model, resolution, window_to_explain=0) (also tested and verified)
