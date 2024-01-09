import pickle

from dater import dater

# Read the windowed anomalous data
file_anomalies = open('pickels/anomaly_data_1.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the windowed background data
file_background = open(f'pickels/background_data_1.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close()

stride = 1
window_size = 32
window_size_med = window_size // 2
window_size_low = window_size_med // 2

med_subwindow_span = window_size - window_size_med
low_subwindow_span = window_size - window_size_low

X = anomalies_windows
# X = background_windows

# print(dater(901, X[0][-1]))
# print(dater(901, X[0][-89]))
# print(dater(901, X[1][-1]))
# print(dater(901, X[1][-1513])) # In one of the big windows, there are 15 of the med ones: 89*17 = 1513

print(dater(901, X[0][-1]))
print(dater(901, X[1][-1]))
print(dater(901, X[1][-17]))
