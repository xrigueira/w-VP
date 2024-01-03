import pickle
import matplotlib.pyplot as plt

# Read the last anomaly data file
file_anomalies = open(f'pickels/anomaly_data_4.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Rename
X = anomalies_windows

stride = 1
num_variables = 6
med_subwindow_span = len(X[1][0]) // (num_variables * stride)
low_subwindow_span = (len(X[0][0])- len(X[2][0])) // (num_variables * stride)

data_high = X[0][0]
data_med = X[1][:(med_subwindow_span + 1)]
data_low = X[2][:(low_subwindow_span + 1)]

import pandas as pd
import numpy as np

# Separate the Series and numpy array parts of your row data
data = pd.read_csv(f'data/labeled_901_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'], index_col=['date'])

data = data.iloc[:, :-2]
# TODO: understand and polish this code. Maybe it needs lower number resolution
series_data = {'ammonium_901': 0.10943396226415096, 'conductivity_901': 0.0830288209606987, 'dissolved_oxygen_901': 0.5890410958904109, 'pH_901': 0.7348484848484849, 'turbidity_901': 0.1626456728462993, 'water_temperature_901': 0.09999999999999998}
array_data = np.array([0.81761006, 0.58323144, 0.20684932, 0.63106061, 0.02381557, 0.59714286])

# Create a boolean mask by checking if each row matches the series_data
mask_series = (data[list(series_data.keys())] == series_data).all(axis=1)

# Create a boolean mask by checking if each row matches the array_data
mask_array = (data[data.columns.difference(series_data.keys())].apply(lambda row: np.array_equal(row.values, array_data), axis=1))

# Combine the two masks
mask = mask_series & mask_array

# Get the index of the row(s) that match the mask
indices = data.index[mask].tolist()

print(indices)  # This will print the index of the row(s) that match row_data