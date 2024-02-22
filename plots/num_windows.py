import pickle
import pandas as pd

# Define main parameters
station = 901
num_iter = 8

num_windows = pd.DataFrame(columns=['iteration', 'resolution', 'background_windows', 'anomalous_windows'])

# Read the testing windowed anomalous data
file_anomalies = open(f'pickels/anomaly_data_0.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the testing windowed background
file_background = open(f'pickels/background_data_0.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close()

num_windows.loc[len(num_windows.index)] = [0, 'high', len(background_windows[0][0]), len(anomalies_windows[0][0])]
num_windows.loc[len(num_windows.index)] = [0, 'med', len(background_windows[0][1]), len(anomalies_windows[0][1])]
num_windows.loc[len(num_windows.index)] = [0, 'low', len(background_windows[0][2]), len(anomalies_windows[0][2])]

for i in range(1, num_iter+1):

    # Read the testing windowed anomalous data
    file_anomalies = open(f'pickels/anomaly_data_{i}.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Read the testing windowed background
    file_background = open(f'pickels/background_data_{i}.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close()
    
    num_windows.loc[len(num_windows.index)] = [0, 'high', len(background_windows[0]), len(anomalies_windows[0])]
    num_windows.loc[len(num_windows.index)] = [0, 'med', len(background_windows[1]), len(anomalies_windows[1])]
    num_windows.loc[len(num_windows.index)] = [0, 'low', len(background_windows[2]), len(anomalies_windows[2])]


num_windows.to_csv(f'plots/num_windows_{station}.csv', sep=',', index=False)