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

window_size_high = 32
window_size_med = 16
window_size_low = 8

# Load models.
iteration = 7

filename = f'models/rf_model_high_{iteration}.sav'
model_high = pickle.load(open(filename, 'rb'))

filename = f'models/rf_model_med_{iteration}.sav'
model_med = pickle.load(open(filename, 'rb'))

filename = f'models/rf_model_low_{iteration}.sav'
model_low = pickle.load(open(filename, 'rb'))

# Read the data: anomalies or background
if data_type == 'anomalies':
    
    # Load the anomalies data
    file_anomalies = open(f'pickels/anomaly_data_test.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Get windowed data and rename it to X
    X = anomalies_windows[0]
    
    lengths = anomalies_windows[-1]
    number_windows_high = [i - window_size_high + 1 for i in lengths]
    number_windows_med = [i - window_size_med + 1 for i in lengths]
    number_windows_low = [i - window_size_low + 1 for i in lengths] 

elif data_type == 'background':
    
    # Load the background data
    file_background = open(f'pickels/background_data_test.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close

    # Get windowed data and rename it to X
    X = background_windows[0]
    
    lengths = background_windows[-1]
    number_windows_high = [i - window_size_high + 1 for i in lengths]
    number_windows_med = [i - window_size_med + 1 for i in lengths]
    number_windows_low = [i - window_size_med + 1 for i in lengths] 

# Find the start and end window index for each resolution and save it in a 2D list
starts_ends = []
for event_number in range(len(number_windows_high)):

    event_start_high = sum(number_windows_high[:event_number]) + event_number
    event_end_high = sum(number_windows_high[:event_number + 1]) + 1 + event_number

    event_start_med = sum(number_windows_med[:event_number]) + event_number
    event_end_med = sum(number_windows_med[:event_number + 1]) + 1 + event_number

    event_start_low = sum(number_windows_low[:event_number]) + event_number
    event_end_low = sum(number_windows_low[:event_number + 1]) + 1 + event_number

    starts_ends.append([[event_start_high, event_end_high], [event_start_med, event_end_med], [event_start_low, event_end_low]])

# Plot a given event
event_number = 0 # anomaly or background event number (anomalies: 4 and 24)

# event_start_high = starts_ends[event_number][0][0]
# event_end_high = starts_ends[event_number][0][1]

# event_data = X[0][event_start_high]
# for i in range(event_start_high + 1, event_end_high):
    
#     # Get the last row of the anomaly
#     last_row = X[0][i][-6:]
    
#     # Add the last row to anomaly_data
#     event_data = np.concatenate((event_data, last_row), axis=0)

# plotter(data=event_data, num_variables=6, windowed=False)

# Get multiresolution windows indixes of the event
event_starts_ends = starts_ends[event_number]

# TODO: modify plotter so it saves the images to a folder. Maybe I can pass the title as a parameter to the function.
# Plot high resolution windows
# plotter(data=X[0][event_starts_ends[0][0]:event_starts_ends[0][1]], num_variables=6, windowed=True)

# Plot medium resolution windows
# plotter(data=X[0][event_starts_ends[1][0]:event_starts_ends[1][1]], num_variables=6, windowed=True)

# Plot low resolution windows
# plotter(data=X[0][event_starts_ends[2][0]:event_starts_ends[2][1]], num_variables=6, windowed=True)

# TODO: Here I would have to build a loop to pass each window_to_explain number to the explainer function
# I don't think it is really efficient because it would have to ready the decision paths each time.
# Get the explainer heatmaps for the high resolution windows
explainer(X[0], model_high, 'high', window_to_explain=0)
