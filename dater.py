import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dater(station, window):

    # Read data
    data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'], index_col=['date'])
    data = data.iloc[:, :-2]

    # Reshape window and define mask
    window = np.array(window).reshape(-1, 6)
    mask = np.zeros(len(data), dtype=bool)
    for window_row in window:
        mask |= (data.values == window_row).all(axis=1)
    
    # Extract dates
    indices = np.where(mask)[0]
    
    date_indices = data.index[indices]
    
    return date_indices

if __name__ == '__main__':

    # Read the last anomaly data file
    file_anomalies = open(f'pickels/anomaly_data_1.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Rename
    X = anomalies_windows

    stride = 1
    num_variables = 6
    med_subwindow_span = len(X[1][0]) // (num_variables * stride)
    low_subwindow_span = (len(X[0][0])- len(X[2][0])) // (num_variables * stride)

    data_high = X[0][-1]
    data_med = X[1][-(med_subwindow_span + 1):]
    data_low = X[2][-(low_subwindow_span + 1):]

    dater(station=901, window=data_med[-1])
    print(data_med[-1])
    dater(station=901, window=data_low[11])
    print(data_low[11])
