import random
import pickle
import numpy as np
import pandas as pd

"""This file outputs three different files:
    1. anomaly_data.npy: contains the data from each anomaly.
    2. background_data.npy: contains 5 times the len(anomaly_data)
    of nonanomalous data also know as background."""

def windower(data, num_variables, window_size, stride):

    """Takes a 2D array with multivariate time series data
    and creates sliding windows. The arrays store the different
    variables in a consecutive manner. E.g. [first 6 variables,
    next 6 variables, and so on]."""
    
    windows = []
    for i in data:
        
        # Get the number of windows
        num_windows = (len(i) - window_size * num_variables) // num_variables + 1
        
        # Create windows
        for j in range(0, num_windows, stride * num_variables):
            window = i[j:j+window_size * num_variables]  # Extract a window of 4 time steps (4 * 6 variables)
            windows.append(window)

    # Convert the result to a NumPy array
    windows = np.array(windows)
    
    return windows

def anomalies(trim_percentage):

    # Load the DataFrame from your dataset
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    
    # Filter the data to select only rows where the label column has a value of 1
    data_anomalies = data[data["label"] == 1]
        
    # Create a new column with the difference between consecutive dates
    date_diff = (data_anomalies['date'] - data_anomalies['date'].shift()).fillna(pd.Timedelta(minutes=15))

    # Create groups of consecutive dates
    date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

    # Get the starting and ending indexes of each group of consecutive dates
    grouped = data.groupby(date_group)
    consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]

    # Trim the start and end of the anomalies
    trimmed_anomalies_indexes = []
    for start, end in consecutive_dates_indexes:
        anomaly_length = end - start
        trim_amount = int(anomaly_length * trim_percentage / 100)
        trimmed_start = start + trim_amount
        trimmed_end = end - trim_amount
        trimmed_anomalies_indexes.append((trimmed_start, trimmed_end))
    
    # Extract the data
    anomaly_data = []
    for start, end in trimmed_anomalies_indexes:
        subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
        anomaly_data.append(subset_rows)
    
    # Save anomalys_data to disk as pickle object
    with open('anomaly_data.pkl', 'wb') as file:
        pickle.dump(anomaly_data, file)
        
    return trimmed_anomalies_indexes

def init_background(anomaly_data, seed, ratio):
    
    # Define random seed
    random.seed(seed)
    
    # Load the DataFrame from your dataset
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

    # Filter the data to select only rows where the label column has a value of 0
    data_background = data[data["label"] == 0]
    
    # Filter the dataset to include only days that meet the ammonium level the condition
    mean_ammonium = np.mean(data_background.ammonium_901)
    data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x['ammonium_901'].max() <= mean_ammonium)
    
    # Extract the length of the anomalies
    len_anomalies = [end - start for start, end in anomaly_data]
    
    # Define background data indexes
    background_indexes = []
    for anomaly_length in len_anomalies:
        if anomaly_length != 0:
            start = random.randint(0, len(data_background) - 1)
            end = start + (anomaly_length * ratio)
            background_indexes.append((start, end))
    
    # Extract the data
    background_data = []
    for start, end in background_indexes:
        
        subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
        background_data.append(subset_rows)
    
    # Save anomalies_data to disk as numpy object
    with open('background_data.pkl', 'wb') as file:
        pickle.dump(background_data, file)
        
    return background_indexes

def background(anomaly_data, background_indexes, seed, ratio, iteration):
    
    # Define random seed
    random.seed(seed)
    
    # Load the DataFrame from your dataset
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    
    # Filter the data to select only rows where the label column has a value of 0
    data_background = data[data["label"] == 0]
    
    # Filter the dataset to include only days that meet the ammonium level the condition
    mean_ammonium = np.mean(data_background.ammonium_901)
    data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x['ammonium_901'].max() <= mean_ammonium)
    
    # Extract the length of the anomalies
    len_anomalies = [end - start for start, end in anomaly_data]
    
    # Define new background data indexes
    new_background_indexes = []
    for anomaly_length in len_anomalies:
        if anomaly_length != 0:
            new_start = random.randint(0, len(data_background) - 1)
            new_end = new_start + (anomaly_length * ratio)
            
            # Check for overlap
            overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
            
            # If there is an overlap, generate a new index
            while overlaps:
                new_start = random.randint(0, len(data_background) - 1)
                new_end = new_start + (anomaly_length * ratio)
                overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
            
            # Append the nonoverlaping indexes to the new list and the old one
            new_background_indexes.append((new_start, new_end))
            background_indexes.append((new_start, new_end))
    
    # Extract the data
    background_data = []
    for start, end in new_background_indexes:
        
        subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
        background_data.append(subset_rows)
    
    # Save anomalies_data to disk as pickle object
    with open(f'background_data_{iteration}.pkl', 'wb') as file:
        pickle.dump(background_data, file)
        
    return background_indexes
    
    # Now I would have to delete the data previously taken out. Remember that you have to add the new indexes
    # of each iteration to the background_indexes list
    
    
if __name__ == '__main__':
    
    anomaly_indexes = anomalies(trim_percentage=10)
    
    background_indexes = init_background(anomaly_indexes, seed=0, ratio=5)
    
    background_indexes = background(anomaly_indexes, background_indexes, seed=0, ratio=5, iteration=1)
    
    background_indexes = background(anomaly_indexes, background_indexes, seed=0, ratio=5, iteration=2)
    