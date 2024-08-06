import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import shap
shap.initjs()
from treeinterpreter import treeinterpreter as ti

from utils import summarizer

"""This program is used to explain the predictions of the model on a particuar event using the treeexplainer and SHAP explainer."""

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

def treexplainer(model, windows, event_number, station, data_type):

    """This function uses the treeexplainer to explain the predictions of the model.
    The function returns the average of the predictions, biases, and contributions for the windows
    of a given event.
    More info here: https://pypi.org/project/treeinterpreter/
    https://blog.datadive.net/interpreting-random-forests/
    ----------
    Arguments:
    model: The trained model to explain its predictions.
    windows: The windows of a given event.
    event_number: The number of the event to explain.
    station: The station number.
    data_type: The type of the data (anomalies or background).

    Returns:
    prediction_means: The average of the predictions for the windows of a given event.
    bias_means: The average of the biases for the windows of a given event.
    contribution_means: The average of the contributions for the windows of a given event.
    """
    
    # Initialize lists to store the results
    predictions = [] # This indicates the predicted probabilities for each class. The first column is the probability for the negative class (normal) and the second column is the probability for the positive class (anomaly)
    biases = [] # This indicates the base value for the prediction before any feature contributions are added. Basically the average of each class in the training set
    contributions = [] # This indicates the contribution of each feature to the prediction. The first column is the contribution for the negative class (normal) and the second column is the contribution for the positive class (anomaly)

    # Loop through the windows to extract the explainability results
    for window in windows:
        
        # Get predictions, biases, and contributions for each sample
        prediction_high, bias_high, contributions_high = ti.predict(model, window.reshape(1, -1))
        predictions.append(prediction_high)
        biases.append(bias_high)
        contributions_high = contributions_high[:, :, 1] # Extract the contributions for the positive class (anomaly). The first column (index 0) is the contribution for the negative class (normal) and the second column (index 1) is the contribution for the positive class (anomaly)
        contributions.append(contributions_high.reshape(-1, 6)) # Group by time window index: -16, -15, ..., +16
    
    # Average the results across the windows to get the final results
    stacked_predictions = np.vstack(predictions)
    prediction_means = np.mean(stacked_predictions, axis=0)
    print('Results for {} number {} of station {}:'.format(data_type, event_number, station))
    print(prediction_means)

    stacked_biases = np.vstack(biases)
    bias_means = np.mean(stacked_biases, axis=0)
    print(bias_means)

    # Calculate the average of each column in the contributions
    stacked_contributions = np.vstack(contributions)
    contribution_means = np.mean(stacked_contributions, axis=0)
    print(contribution_means)

def shap_analysis(model, windows, event_number, station, data_type, summarized=True):
    
    """This function uses the SHAP explainer to explain the predictions of the model.
    ----------
    Arguments:
    model: The trained model to explain its predictions.
    windows: The windows of a given event.
    summarized: A boolean to indicate if the SHAP values should be summarized or not.

    Returns:
    None
    """

    # Define feature names for all resolution levels
    feature_names_high = [
                    'am-16', 'co-16', 'do-16', 'ph-16', 'tu-16', 'wt-16',
                    'am-15', 'co-15', 'do-15', 'ph-15', 'tu-15', 'wt-15',
                    'am-14', 'co-14', 'do-14', 'ph-14', 'tu-14', 'wt-14',
                    'am-13', 'co-13', 'do-13', 'ph-13', 'tu-13', 'wt-13',
                    'am-12', 'co-12', 'do-12', 'ph-12', 'tu-12', 'wt-12',
                    'am-11', 'co-11', 'do-11', 'ph-11', 'tu-11', 'wt-11',
                    'am-10', 'co-10', 'do-10', 'ph-10', 'tu-10', 'wt-10',
                    'am-9', 'co-9', 'do-9', 'ph-9', 'tu-9', 'wt-9',
                    'am-8', 'co-8', 'do-8', 'ph-8', 'tu-8', 'wt-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'tu-7', 'wt-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'tu-6', 'wt-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'tu-5', 'wt-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'tu+5', 'wt+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'tu+6', 'wt+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'tu+7', 'wt+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'tu+8', 'wt+8',
                    'am+9', 'co+9', 'do+9', 'ph+9', 'tu+9', 'wt+9',
                    'am+10', 'co+10', 'do+10', 'ph+10', 'tu+10', 'wt+10',
                    'am+11', 'co+11', 'do+11', 'ph+11', 'tu+11', 'wt+11',
                    'am+12', 'co+12', 'do+12', 'ph+12', 'tu+12', 'wt+12',
                    'am+13', 'co+13', 'do+13', 'ph+13', 'tu+13', 'wt+13',
                    'am+14', 'co+14', 'do+14', 'ph+14', 'tu+14', 'wt+14',
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt+15',
                    'am+16', 'co+16', 'do+16', 'ph+16', 'tu+16', 'wt+16'
                    ]
    
    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Get the SHAP values for the windows
    shap_values = explainer.shap_values(np.array(windows))

    if summarized:
        
        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        # Decision plot for the summarized version
        plt.sca(axs)
        shap.decision_plot(explainer.expected_value[1],
                        summarizer(shap_values[1], num_variables=6), 
                        feature_names=['am', 'co', 'do', 'ph', 'tu', 'wt'],
                        plot_color='coolwarm',
                        title='SHAP decision plot',
                        show=False,
                        auto_size_plot=False)
        
        # Adjust layout and save the figure
        plt.savefig(f'results/decision_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')
        
        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))

        # Summary plot for the summarized version
        plt.sca(axs)
        shap.summary_plot(summarizer(shap_values[1], num_variables=6),
                        summarizer(windows, num_variables=6),
                        # plot_type='violin',
                        feature_names=['am', 'co', 'do', 'ph', 'tu', 'wt'],
                        cmap='coolwarm',
                        show=False,
                        plot_size=None)
        
        plt.title('SHAP beeswarm plot')
        
        # Adjust layout and save the figure
        plt.savefig(f'results/beeswarm_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')

        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        
        # Bar plot for the summarized version
        plt.sca(axs)
        shap.summary_plot(summarizer(shap_values[1], num_variables=6),
                        summarizer(windows, num_variables=6), 
                        feature_names=['am', 'co', 'do', 'ph', 'tu', 'wt'],
                        plot_type='bar',
                        color='indianred',
                        show=False,
                        plot_size=None)
        
        # Change the title of the x-axis
        plt.xlabel('Mean(|SHAP value|)')
        plt.title('SHAP bar plot')
        
        # Adjust layout and save the figure
        plt.savefig(f'results/bar_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')
    
    else:

        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        
        # Decision plot for the selected sample
        plt.sca(axs)
        shap.decision_plot(explainer.expected_value[1], 
                        shap_values[1],
                        feature_names=feature_names_high,
                        plot_color='coolwarm',
                        title='Shape decision plot')

        # Adjust layout and save the figure
        plt.savefig(f'results/decision_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')

        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        
        # Summary plot for the selected sample
        plt.sca(axs)
        shap.summary_plot(shap_values[1],
                        feature_names=feature_names_high,
                        # plot_type='violin',
                        title='Shape summary plot',
                        cmap='coolwarm')
        
        plt.title('SHAP beeswarm plot')
        
        # Adjust layout and save the figure
        plt.savefig(f'results/beeswarm_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')

        # Creat a figure
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        
        # Bar plot for the selected sample
        plt.sca(axs)
        shap.summary_plot(shap_values[1],
                        feature_names=feature_names_high,
                        plot_type='bar',
                        title='Shape summary plot',
                        color='indianred')
        
        # Change the title of the x-axis
        plt.xlabel('Mean(|SHAP value|)')
        plt.title('SHAP bar plot')
        
        # Adjust layout and save the figure
        plt.savefig(f'results/bar_plot_{station}_{data_type[:2]}_{event_number}.pdf', dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    station = 901

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
    
    #%% Get the results for the labeled anomalies
    for event_number_main in anomalies_events:
        if event_number_main == 1:
            logging.info('Processing anomaly event number %d', event_number_main)

            # Update data type, starts_ends and X
            data_type = 'anomalies'
            starts_ends = starts_ends_anomalies
            X = X_anomalies

            resolution = 0 # High resolution = 0, medium resolution = 1, low resolution = 2
            windows = X[resolution][starts_ends[event_number_main][resolution][0]:starts_ends[event_number_main][resolution][1]]

            # Explain the predictions of the model with the treeexplainer
            treexplainer(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2])

            # Explain the predictions of the model with the SHAP explainer
            shap_analysis(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2], summarized=True)
    
    #%% Get the results for the detected anomalies
    for event_number_main in background_anomalies_events:
        if event_number_main == 42:
            logging.info('Processing detected anomaly event number %d', event_number_main)
            
            # Update data type, starts_ends and X
            data_type = 'background'
            starts_ends = starts_ends_background
            X = X_background

            resolution = 0 # High resolution = 0, medium resolution = 1, low resolution = 2
            windows = X[resolution][starts_ends[event_number_main][resolution][0]:starts_ends[event_number_main][resolution][1]]

            # Explain the predictions of the model with the treeexplainer
            treexplainer(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2])

            # Explain the predictions of the model with the SHAP explainer
            shap_analysis(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2], summarized=True)

    #%% Get the results for the true background events
    for event_number_main in background_background_events:
        if event_number_main == 25:
            logging.info('Processing true background event number %d', event_number_main)

            # Update data type, starts_ends and X
            data_type = 'background'
            starts_ends = starts_ends_background
            X = X_background

            resolution = 0 # High resolution = 0, medium resolution = 1, low resolution = 2
            windows = X[resolution][starts_ends[event_number_main][resolution][0]:starts_ends[event_number_main][resolution][1]]

            # Explain the predictions of the model with the treeexplainer
            treexplainer(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2])

            # Explain the predictions of the model with the SHAP explainer
            shap_analysis(model_high, windows, event_number=event_number_main, station=station, data_type=data_type[:2], summarized=True)

