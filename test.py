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
# lengths = anomalies_windows[-1]
# lengths = background_windows[-1]
# number_windows = [i - window_size + 1 for i in lengths]


# print(dater(901, X[0][-1]))
# print(dater(901, X[0][-89]))
# print(dater(901, X[1][-1]))
# print(dater(901, X[1][-1513])) # In one of the big windows, there are 15 of the med ones: 89*17 = 1513

print(dater(901, X[0][-1]))
print(dater(901, X[1][-1]))
print(dater(901, X[1][-17]))

# Actual loop for testing purposes. This is from another test file that I moved here
X = anomalies_windows[0]
lengths = anomalies_windows[-1]
number_windows = [i - window_size + 1 for i in lengths]
med_subwindow_span = window_size - window_size_med
low_subwindow_span = window_size - window_size_low

index_high = 0
start_index_med, end_index_med = 0, med_subwindow_span 
start_index_low, end_index_low = 0, low_subwindow_span

counter_number_windows = 0
current_window_number = 0
for i in range(len(X[0])):
    print(i)
    print(start_index_med, end_index_med + 1)
    print(current_window_number, number_windows[counter_number_windows])
    # print('High', dater(901, X[0][index_high]))
    # print('Med', dater(901, X[1][start_index_med:end_index_med + 1]))
    print('------------------')
    if current_window_number == number_windows[counter_number_windows]:
        index_high = index_high + stride
        start_index_med, end_index_med = end_index_med + 1, end_index_med + med_subwindow_span + 1
        start_index_low, end_index_low = end_index_low + 1, end_index_low + low_subwindow_span + 1
        counter_number_windows += 1
        current_window_number = 0
    else:
        index_high = index_high + stride
        start_index_med, end_index_med = start_index_med + stride, end_index_med + stride
        start_index_low, end_index_low = start_index_low + stride, end_index_low + stride
        current_window_number += 1

    if i == 15:
        break


