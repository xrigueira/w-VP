import random
import pickle
import numpy as np
import pandas as pd

"""This file outputs two different files:
    1. anomaly_data.npy: contains the data from each anomaly.
    2. background_data.npy: contains 5 times the len(anomaly_data)
    of nonanomalous data also know as background."""

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

def background(anomaly_data, seed, ratio):
    
    # Define random seed
    random.seed(seed)
    
    # Load the DataFrame from your dataset
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

    # Extract the length of the anomalies
    len_anomalies = [end - start for start, end in anomaly_data]
    
    # Filter the data to select only rows where the label column has a value of 0
    data_background = data[data["label"] == 0]
    
    # Filter the dataset to include only days that meet the ammonium level the condition
    mean_ammonium = np.mean(data_background.ammonium_901)
    filtered_df = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x['ammonium_901'].max() <= mean_ammonium)
    
    # Define background data indexes
    background_indexes = []
    for anomaly_length in len_anomalies:
        if anomaly_length != 0:
            start = random.randint(0, len(filtered_df))
            end = start + (anomaly_length * ratio)
            background_indexes.append((start, end))
    
    # Extract the data
    background_data = []
    for start, end in background_indexes:
        
        subset_rows = filtered_df.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
        background_data.append(subset_rows)
    
    # Save anomalies_data to disk as numpy object
    with open('background_data.pkl', 'wb') as file:
        pickle.dump(background_data, file)
        
if __name__ == '__main__':
    
    anomaly_indexes = anomalies(trim_percentage=10)
    
    background(anomaly_indexes, seed=0, ratio=5)