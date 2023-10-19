import pickle
import numpy as np

# Read the windowed anomalous data
# file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
# anomalies_windows = pickle.load(file_anomalies)
# file_anomalies.close()

file_anomalies = open('pickels/background_data_0.pkl', 'rb')
background_windows = pickle.load(file_anomalies)
file_anomalies.close()

score_Xs_high = np.load('score_Xs_high.npy', allow_pickle=False, fix_imports=False)
score_Xs_med = np.load('score_Xs_med.npy', allow_pickle=False, fix_imports=False)
score_Xs_low = np.load('score_Xs_low.npy', allow_pickle=False, fix_imports=False)

# Now lets try to get the scores for the first high window across all levels
stride = 1
num_variables = 6
med_subwindow_span = len(background_windows[1][0]) // (num_variables * stride)
low_subwindow_span = (len(background_windows[0][0])- len(background_windows[2][0])) // (num_variables * stride)

index_high = 0
start_index_med, end_index_med = 0, med_subwindow_span + 1
start_index_low, end_index_low = 0, low_subwindow_span + 1

for i in range(len(score_Xs_high)):
    
    scores_high = score_Xs_high[index_high]
    scores_med = score_Xs_med[start_index_med:end_index_med]
    scores_low = score_Xs_low[start_index_low:end_index_low]
    
    # See how to combine the scores
    # If they are over a certain value I would have to save their indexes
    # to get their windows later and just add them to their respective window
    # length
    
    # Update the index values
    index_high = index_high + stride
    start_index_med, end_index_med = start_index_med + stride, end_index_med + stride
    start_index_low, end_index_low = start_index_low + stride, end_index_low + stride

# # Next do the second iteration
# print(score_Xs_high[0+stride])
# print(len(score_Xs_med[0+stride:med_subwindow_span+stride+1]))
# print(len(score_Xs_low[0+stride:low_subwindow_span+stride+1]))

# I would have to start the indixes before the loop and add the stride
# at the end of each iteration
