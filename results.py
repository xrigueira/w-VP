import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree

from utils import dater
from utils import plotter
from utils import explainer
from utils import mean_plotter

def plotter_all(event_number):

    event_start_high = starts_ends[event_number][0][0]
    event_end_high = starts_ends[event_number][0][1]

    event_data = X[0][event_start_high]
    for i in range(event_start_high + 1, event_end_high):
        
        # Get the last row of the anomaly
        last_row = X[0][i][-6:]
        
        # Add the last row to anomaly_data
        event_data = np.concatenate((event_data, last_row), axis=0)

    plotter(data=event_data, num_variables=6, legend=True, name=f'event_{event_number}')

def get_results(event_number):

    event_start_high = starts_ends[event_number][0][0]
    event_end_high = starts_ends[event_number][0][1]

    event_data = X[0][event_start_high]
    for i in range(event_start_high + 1, event_end_high):
        
        # Get the last row of the anomaly
        last_row = X[0][i][-6:]
        
        # Add the last row to anomaly_data
        event_data = np.concatenate((event_data, last_row), axis=0)

    plotter(data=event_data, num_variables=6, legend=True, name=f'event_{event_number}')

    # Get multiresolution windows indixes of the event
    event_starts_ends = starts_ends[event_number]
    
    # Plot, explan and get mean for high resolution windows
    for window_num, window in enumerate(X[0][event_starts_ends[0][0]:event_starts_ends[0][1]]):

        plotter(data=window, num_variables=6, legend=False, name=f'event_{event_number}_high_{window_num}')
        explainer(data=window, model=model_high, resolution='high', name=f'event_{event_number}_high_{window_num}')
        mean_plotter(data=window, resolution='high', num_variables=6, name=f'event_{event_number}_high_{window_num}')

    # Plot, explan and get mean for medium resolution windows
    for window_num, window in enumerate(X[1][event_starts_ends[1][0]:event_starts_ends[1][1]]):
        
        plotter(data=window, num_variables=6, legend=False, name=f'event_{event_number}_med_{window_num}')
        explainer(data=window, model=model_med, resolution='med', name=f'event_{event_number}_med_{window_num}')
    
    # Plot, explan and get mean for low resolution windows
    for window_num, window in enumerate(X[2][event_starts_ends[2][0]:event_starts_ends[2][1]]):
        
        plotter(data=window, num_variables=6, legend=False, name=f'event_{event_number}_low_{window_num}')
        explainer(data=window, model=model_low, resolution='low', name=f'event_{event_number}_low_{window_num}')

def majority_vote(data_high, data_med, data_low):
        
    vote_high = round(sum(data_high) / len(data_high))
    vote_med = round(sum(data_med) / len(data_med))
    vote_low = round(sum(data_low) / len(data_low))
    
    if sum([vote_high, vote_med, vote_low]) >= 2:
        return 1
    else:
        return 0

if __name__ == '__main__':

    data_type = 'background' # 'anomalies' or 'background

    window_size_high, window_size_med, window_size_low = 32, 16, 8

    # Load models
    iteration = 5

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
    
    plot_all = str(input('Plot all events? (y/n): '))
    if plot_all == 'y':
        for event_number in range(len(number_windows_high)):
            plotter_all(event_number)

    if data_type == 'anomalies':
        
        get_results(event_number=4) # 901: anomalies 4 or 24
                                    # 905: anomalies 18 or 33
                                    # 906: anomalies 1 or 3
                                    # 907: anomalies 25 or 34

    elif data_type == 'background':
        
        # Read background results
        y_hats_high = np.load('preds/y_hats_high.npy', allow_pickle=False, fix_imports=False)
        y_hats_med = np.load('preds/y_hats_med.npy', allow_pickle=False, fix_imports=False)
        y_hats_low = np.load('preds/y_hats_low.npy', allow_pickle=False, fix_imports=False)
        
        # Get multiresolution votes for each event
        votes = [majority_vote(y_hats_high[i[0][0]:i[0][1]], y_hats_med[i[1][0]:i[1][1]], y_hats_low[i[2][0]:i[2][1]]) for i in starts_ends]

        anomalies_events = np.where(np.array(votes) == 1)[0]
        background_events = np.where(np.array(votes) == 0)[0]

        event_type = input('Choose event type (anomalies (1) or background (0)): ')

        if event_type == '1':
            event_number = int(input(f'Choose an event number {anomalies_events}: '))
        elif event_type == '0':
            event_number = int(input(f'Choose an event number {background_events}: '))

    #     get_results(event_number=event_number) # 901: 0 for true background, 28 for idenfitied anomaly
    #                                             # 905: 8 for true background, 21 for identified anomaly
    #                                             # 906: 1 for everything?
    #                                             # 907: 25 for true background, 16 for identified anomaly