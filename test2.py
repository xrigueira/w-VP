import pickle

from dater import dater

# Read the windowed anomalous data
file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the windowed background data
file_background = open(f'pickels/background_data_0.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close()

X = background_windows
X = anomalies_windows

index = 14
print(X[0][index])
print(X[1][index:index+17][0])
print(X[1][index:index+17][-1])
print(dater(901, X[0][index]))
print(dater(901, X[1][index:index+17]))
