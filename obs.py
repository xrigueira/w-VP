"""Contains obsolete methods just in case"""

# Original windower. Not multiresolution
# def windower(self, data):

#     """Takes a 2D array with multivariate time series data
#     and creates sliding windows. The arrays store the different
#     variables in a consecutive manner. E.g. [first 6 variables,
#     next 6 variables, and so on].
#     ----------
#     Arguments:
#     data (pickle): file with the time-series data to turn 
#     into windows.
#     num_variables (int): the number of variables in the data.
#     window_size (int): the size of the windows.
#     stride (int): the stride of the windows.
    
#     Returns:
#     windows (np.array): time series data grouped in windows."""
    
#     windows = []
#     for i in data:
        
#         # Get the number of windows
#         num_windows = (len(i) - self.window_size * self.num_variables) // self.num_variables + 1
        
#         # Create windows
#         for j in range(0, num_windows, self.stride * self.num_variables):
#             window = i[j:j+self.window_size * self.num_variables]
#             windows.append(window)

#     # Convert the result to a NumPy array
#     windows = np.array(windows)
    
#     return windows

# Multiresolution windower test code
# import numpy as np

# data = [[*range(1, 65, 1)]]

# def windower(data, window_size, stride, num_variables):

#     windows = []
#     if window_size > 1:
#         for i in data:
#             num_windows = (len(i) - window_size * num_variables) // (stride * num_variables) + 1
#             for j in range(0, num_windows, stride):
#                 window = i[j * num_variables: (j * num_variables) + (window_size * num_variables)]
#                 windows.append(window)
#         window_size = window_size // 2
#         return [windows] + windower(data, window_size, stride, num_variables)
#     else:
#         return []

# window_size = 8
# num_variables = 6
# stride = 1

# windows = windower(data, window_size, stride, num_variables)

# Multiresolution combinator test code
# import pickle
# import numpy as np

# # Perform majority voting
# def majority_vote(*args):
#     num_ones = sum(1 for result in args if result >= 0.51)
#     return int(num_ones > len(args) / 2)

# # Read the windowed anomalous data
# file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
# anomalies_windows = pickle.load(file_anomalies)
# file_anomalies.close()

# file_anomalies = open('pickels/background_data_0.pkl', 'rb')
# background_windows = pickle.load(file_anomalies)
# file_anomalies.close()

# score_Xs_high = np.load('score_Xs_high.npy', allow_pickle=False, fix_imports=False)
# score_Xs_med = np.load('score_Xs_med.npy', allow_pickle=False, fix_imports=False)
# score_Xs_low = np.load('score_Xs_low.npy', allow_pickle=False, fix_imports=False)

# # Now lets try to get the scores for the first high window across all levels
# stride = 1
# num_variables = 6
# med_subwindow_span = len(background_windows[1][0]) // (num_variables * stride)
# low_subwindow_span = (len(background_windows[0][0])- len(background_windows[2][0])) // (num_variables * stride)

# index_high = 0
# start_index_med, end_index_med = 0, med_subwindow_span + 1
# start_index_low, end_index_low = 0, low_subwindow_span + 1

# indexes_anomalies_windows_high, indexes_background_windows_high = [], []
# indexes_anomalies_windows_med, indexes_background_windows_med = [], []
# indexes_anomalies_windows_low, indexes_background_windows_low = [], []
# for i in range(len(score_Xs_high)):
    
#     scores_high = score_Xs_high[index_high]
#     scores_med = score_Xs_med[start_index_med:end_index_med]
#     scores_low = score_Xs_low[start_index_low:end_index_low]
    
#     # Combine the float result with the majority voting of the lists
#     multiresolution_vote = majority_vote(scores_high, *scores_med, *scores_low)
    
#     if multiresolution_vote == 1:
#         indexes_anomalies_windows_high.append(index_high)
#         indexes_anomalies_windows_med.append((start_index_med, end_index_med))
#         indexes_anomalies_windows_low.append((start_index_low, end_index_low))
#     else:
#         indexes_background_windows_high.append(index_high)
#         indexes_background_windows_med.append((start_index_med, end_index_med))
#         indexes_background_windows_low.append((start_index_low, end_index_low))
    
#     # Update the index values
#     index_high = index_high + stride
#     start_index_med, end_index_med = start_index_med + stride, end_index_med + stride
#     start_index_low, end_index_low = start_index_low + stride, end_index_low + stride
    
# print(indexes_anomalies_windows_low)