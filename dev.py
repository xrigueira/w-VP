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

# Plot a given anomaly
event_number = 5 # anomaly or background event number

event_start = sum(number_windows[:event_number]) + event_number
event_end = sum(number_windows[:event_number + 1]) + 1 + event_number

event_data = X[event_start]
for i in range(event_start + 1, event_end):
    
    # Get the last row of the anomaly
    last_row = X[i][-6:]
    
    # Add the last row to anomaly_data
    event_data = np.concatenate((event_data, last_row), axis=0)

# plotter(data=event_data, num_variables=6, windowed=False)

# Explain the given anomaly
# 1. For a given anomaly I have to get the indices of the windows at all resolutions. This can be done based on how main gets the indices.
# 2. Once I have the indices, I can select the windows I want to plot and explain

X = X[:20] # Subset the data to the first 20 anomalies just for testing purposes
explainer(X, model, resolution, window_to_explain=0)
