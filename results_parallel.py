
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import dater, event_plotter, depths, attention, multivariate_attention, kl_divergence
from utils import attention_plotter, multivariate_attention_plotter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_starts_ends(num_win_high, num_win_med, num_win_low):

    """Extacts the start and end window index for each resolution and save it in a 2D list"""

    starts_ends = []
    for event_number in range(len(num_win_high)):

        event_start_high = sum(num_win_high[:event_number]) + event_number
        event_end_high = sum(num_win_high[:event_number + 1]) + 1 + event_number

        event_start_med = sum(num_win_med[:event_number]) + event_number
        event_end_med = sum(num_win_med[:event_number + 1]) + 1 + event_number

        event_start_low = sum(num_win_low[:event_number]) + event_number
        event_end_low = sum(num_win_low[:event_number + 1]) + 1 + event_number

        starts_ends.append([[event_start_high, event_end_high], [event_start_med, event_end_med], [event_start_low, event_end_low]])

    return starts_ends

def majority_vote(high, med, low):
    
    vote_high = sum(high) / len(high)
    vote_med = sum(med) / len(med)
    vote_low = sum(low) / len(low)

    if (1/3 * vote_high + 1/3 * vote_med + 1/3 * vote_low) >= 0.9:
        return 1
    elif (1/3 * vote_high + 1/3 * vote_med + 1/3 * vote_low) <= 0.1:
        return 0

# For parallel processing
def calculate_kl_divergence(starts_ends, X, events, models, attention_multivariate):
    kl_distances = []
    for event_number in events:
        variables_depth, max_depth = depths(starts_ends, X, models=models, event_number=event_number)
        attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt = attention(variables_depth, max_depth)
        attention_multivariate_compare = multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt)
        kl_distance = kl_divergence(attention_multivariate, attention_multivariate_compare)
        kl_distances.append(kl_distance)
    return kl_distances

def parallel_kl_divergence():
    with ProcessPoolExecutor() as executor:
        futures = []
        # For anomalies
        futures.append(executor.submit(calculate_kl_divergence, starts_ends_anomalies, X_anomalies, anomalies_events, [model_high, model_med, model_low], attention_multivariate))
        # For anomalous background
        futures.append(executor.submit(calculate_kl_divergence, starts_ends_background, X_background, background_anomalies_events, [model_high, model_med, model_low], attention_multivariate))
        # For true background
        futures.append(executor.submit(calculate_kl_divergence, starts_ends_background, X_background, background_background_events, [model_high, model_med, model_low], attention_multivariate))

        for i, future in enumerate(as_completed(futures)):
            kl_distances[i] = future.result()
            logging.info(f'Finished kl distances with {["anomalies", "anomalous background", "background"][i]}')

if __name__ == '__main__':

    station = 901
    data_type = 'anomalies' # 'anomalies' or 'background'. See also line 127

    window_size_high, window_size_med, window_size_low = 32, 16, 8

    # Load models
    iteration = 9

    filename = f'models/rf_model_high_{iteration}.sav'
    model_high = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_med_{iteration}.sav'
    model_med = pickle.load(open(filename, 'rb'))

    filename = f'models/rf_model_low_{iteration}.sav'
    model_low = pickle.load(open(filename, 'rb'))

    # Load the anomalies data
    file_anomalies = open(f'pickels/anomaly_data_pred.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()
    
    # Load the background data
    file_background = open(f'pickels/background_data_pred.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close()

    # Get windowed data and rename it to X_anomalies and X_background
    X_anomalies = anomalies_windows[0]
    X_background = background_windows[0]

    # Get the lengths of the windows
    lengths_anomalies = anomalies_windows[-1]
    lengths_background = background_windows[-1]

    # Get the number of windows for each resolution, anomalies and background
    number_windows_high_anomalies = [i - window_size_high + 1 for i in lengths_anomalies]
    number_windows_med_anomalies = [i - window_size_med + 1 for i in lengths_anomalies]
    number_windows_low_anomalies = [i - window_size_low + 1 for i in lengths_anomalies]

    number_windows_high_background = [i - window_size_high + 1 for i in lengths_background]
    number_windows_med_background = [i - window_size_med + 1 for i in lengths_background]
    number_windows_low_background = [i - window_size_low + 1 for i in lengths_background]

    # Find the start and end window index for each resolution and save it in a 2D list
    starts_ends_anomalies = get_starts_ends(number_windows_high_anomalies, number_windows_med_anomalies, number_windows_low_anomalies)
    starts_ends_background = get_starts_ends(number_windows_high_background, number_windows_med_background, number_windows_low_background)
    
    # Read background results
    y_hats_high = np.load('preds/y_hats_high.npy', allow_pickle=False, fix_imports=False)
    y_hats_med = np.load('preds/y_hats_med.npy', allow_pickle=False, fix_imports=False)
    y_hats_low = np.load('preds/y_hats_low.npy', allow_pickle=False, fix_imports=False)
        
    # Get multiresolution votes for each background event
    votes = [majority_vote(y_hats_high[i[0][0]:i[0][1]], y_hats_med[i[1][0]:i[1][1]], y_hats_low[i[2][0]:i[2][1]]) for i in starts_ends_background]

    # Ge the index of the true anomalies and background events which the background set
    background_anomalies_events = np.where(np.array(votes) == 1)[0]
    background_background_events = np.where(np.array(votes) == 0)[0]
    print('Background anomalies:', background_anomalies_events)
    print('True background:', background_background_events)

    # Get the number of actual anomalous events
    anomalies_events = range(len(number_windows_high_anomalies))
    
    for event_number_main in anomalies_events: # This has to be changed when switching from anomalies to background
        logging.info('Processing event number %d', event_number_main)

        if data_type == 'anomalies':
            starts_ends = starts_ends_anomalies
            X = X_anomalies
        
        elif data_type == 'background':
            starts_ends = starts_ends_background
            X = X_background

        # Plot the event
        event_plotter(starts_ends, X, event_number_main, station=station, type=data_type[:2])
        logging.info('Finished event plot')

        # Get the depths of the variables
        variables_depth, max_depth = depths(starts_ends, X, models=[model_high, model_med, model_low], event_number=event_number_main)

        # Get the attention maps
        attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt = attention(variables_depth, max_depth)
        
        # Get the multivariate attention map
        attention_multivariate = multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt)

        # Plot the attention
        attention_maps = [attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt]
        attention_plotter(attention_maps, event_number=event_number_main, station=station, type=data_type[:2])

        # Plot the multivariate attention
        multivariate_attention_plotter(attention_multivariate, event_number=event_number_main, station=station, type=data_type[:2])
        logging.info('Finished attention maps')

        # Store the Kullback-Leibler divergence between the selected event and the anomalies
        kl_distances = [[], [], []]

        parallel_kl_divergence()

        # Plot the KL divergences
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(kl_distances[0], color='lightcoral', label='True anomalies', linewidth=1, fill=True, ax=ax)
        sns.kdeplot(kl_distances[1], color='limegreen', label='Anomalous background', linewidth=1, fill=True, ax=ax)
        sns.kdeplot(kl_distances[2], color='cornflowerblue', label='True background', linewidth=1, fill=True, ax=ax)

        ax.set_title('Kullback-Leibler divergence distributions', fontfamily='serif', fontsize=20)
        ax.set_xlabel('Divergence', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(loc='upper left', fontsize=16)
        # plt.show()

        plt.savefig(f'results/kl_divergence_{station}_{data_type[:2]}_{event_number_main}.pdf', format='pdf', dpi=300, bbox_inches='tight')
