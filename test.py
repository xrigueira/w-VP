import pickle

# Read the windowed anomalous data
file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the windowed anomalous data
file_anomalies = open('pickels/anomaly_data_01.pkl', 'rb')
mr_anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

print((anomalies_windows[0]))
print((mr_anomalies_windows[1][0]))