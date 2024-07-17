
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree

from utils import dater
from utils import plotter
from utils import explainer

def plotter_all(starts_ends, X, event_number, station):

    event_start_high = starts_ends[event_number][0][0]
    event_end_high = starts_ends[event_number][0][1]

    event_data = X[0][event_start_high]
    for i in range(event_start_high + 1, event_end_high):
        
        # Get the last row of the anomaly
        last_row = X[0][i][-6:]
        
        # Add the last row to anomaly_data
        event_data = np.concatenate((event_data, last_row), axis=0)

    plotter(data=event_data, num_variables=6, station=station, legend=True, name=f'event_{event_number}')

if __name__ == '__main__':

    station = 901
    data_type = 'anomalies' # 'anomalies' or 'background

    window_size_high, window_size_med, window_size_low = 32, 16, 8

    # Load models
    iteration = 9

    filename = f'models/rf_model_high_{iteration}.sav'
    model_high = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_med_{iteration}.sav'
    model_med = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_low_{iteration}.sav'
    model_low = pickle.load(open(filename, 'rb'))

    # Read the data: anomalies or background
    if data_type == 'anomalies':
        
        # Load the anomalies data
        file_anomalies = open(f'pickels/anomaly_data_pred.pkl', 'rb')
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
        file_background = open(f'pickels/background_data_pred.pkl', 'rb')
        background_windows = pickle.load(file_background)
        file_background.close

        # Get windowed data and rename it to X
        X = background_windows[0]
        
        lengths = background_windows[-1]
        number_windows_high = [i - window_size_high + 1 for i in lengths]
        number_windows_med = [i - window_size_med + 1 for i in lengths]
        number_windows_low = [i - window_size_low + 1 for i in lengths] 

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
    
    # plot_all = str(input('Plot all events? (y/n): '))
    # if plot_all == 'y':
    #     for event_number in range(len(number_windows_high)):
    #         plotter_all(starts_ends, X, event_number, station=station)
    
    event_number = 0

    # Get a explainer plot
    variables_depth, max_depth = explainer(starts_ends, X, models=[model_high, model_med, model_low], event_number=event_number)

    # Get the number of times each variable appears at each depth
    frequency_depth = np.zeros((len(variables_depth), max_depth))
    for i, var in enumerate(variables_depth.keys()):
        for pos in variables_depth[var]:
            frequency_depth[i, pos] += 1

    # Convert data to int
    frequency_depth = frequency_depth.astype(int) 

    # Assign the data to the dictionary with the same keys
    attention = {}
    for var in variables_depth.keys():
        attention[var] = []
    
    for i, var in enumerate(variables_depth.keys()):
        for j in range(max_depth):
            attention[var].append(frequency_depth[i, j])

    # Subset a specific variable
    variable = 'am'
    heatmap_am = {key: value for key, value in attention.items() if key.startswith(variable)}

    # Plot the heatmap
    sns.heatmap(pd.DataFrame(heatmap_am).T, cmap='coolwarm', cbar=True)
    plt.xlabel('Depth')
    plt.ylabel('Variable')
    plt.title(f'Attention map for variable {variable}')
    plt.tight_layout()
    plt.show()

    # # Subset a specific variable
    # variable = 'co'
    # subset_variables_frequency = {key: value for key, value in attention.items() if key.startswith(variable)}

    # # Turn into a 2D numpy array
    # heatmap_co = np.array(list(subset_variables_frequency.values()))

    # # Plot the heatmap
    # yticks = [var[2:] for var in subset_variables_frequency.keys()]
    # sns.heatmap(heatmap_co, yticklabels=yticks, cmap='YlOrRd', cbar=True)
    # plt.xlabel('Depth')
    # plt.ylabel('Variable')
    # plt.title(f'Attention map for variable {variable}')
    # # plt.show()

    # # Normalize the heatmaps to ensure they sum to 1
    # heatmap_am = heatmap_am / np.sum(heatmap_am)
    # heatmap_co = heatmap_co / np.sum(heatmap_co)

    # # A small value epsilon is added
    # epsilon = np.finfo(float).eps
    # heatmap_am = heatmap_am + epsilon
    # heatmap_co = heatmap_co + epsilon

    # # Compute KL divergence
    # kl_divergence = np.sum(heatmap_am * np.log(heatmap_am / heatmap_co))

    # print(f"KL Divergence: {kl_divergence}")
