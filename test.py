import pickle

# Read the background data file
file_background = open(f'pickels/background_data_4.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close()

# Rename
X = background_windows

stride = 1
num_variables = 6
med_subwindow_span = len(X[1][0]) // (num_variables * stride)
low_subwindow_span = (len(X[0][0])- len(X[2][0])) // (num_variables * stride)

data_high = X[0][-1]
data_med = X[1][-(med_subwindow_span + 1):]
data_low = X[2][-(low_subwindow_span + 1):]

print(data_high)
print(data_med[-1])
print(data_low[-1])