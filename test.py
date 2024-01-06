import pickle
import pandas as pd

from dater import dater

def windower(data, window_size, stride, num_variables):
        
    """
    Takes a 2D list of NumPy arrays with multivariate 
    time series data and creates multiresolution sliding 
    windows. The size of the slinding windows gets halved 
    each time. The resulting windows store different 
    variables in a consecutive manner. E.g. [first 6 variables, 
    next 6 variables, and so on].
    ----------
    Arguments:
    data (list): file with the time-series data to turn 
    into windows.
    num_variables (int): the number of variables in the data.
    window_size (int): the size of the windows.
    stride (int): the stride of the windows.
    
    Returns:
    windows (list): time series data grouped in windows"""
    
    windows = []
    if window_size > 4: # This way the maximum window would be 8 data points
        
        for i in data:
            
            # Get the number of windows
            num_windows = (len(i) - window_size * num_variables) // (stride * num_variables) + 1
            
            # Create the windows
            for j in range(0, num_windows, stride):
                window = i[j * num_variables: (j * num_variables) + (window_size * num_variables)]
                windows.append(window)
        
        window_size = window_size // 2 # Halve window size
        
        return [windows] + windower(data, window_size, stride, num_variables)  # Recursive call
    
    else:
        
        # Restore the window size after recursion
        window_size = window_size
        
        return []

def anomalies(station, trim_percentage, window_size, stride, num_variables):
        
    """Extracts the anomalies from the database and
    saves them to a pickle file.
    ----------
    Arguments:
    self.
    
    Stores:
    anomaly_data (pickle): file with the multivariate data.
    from each anomaly.
    
    Returns:
    trimmed_anomalies_indexes (list): start and end indexes of the extracted
    anomaly data.
    """
    
    # Load the data
    data = pd.read_csv(f'data/labeled_{station}_smo_test.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    data_copy = data.copy()

    # Create a new column 'group' that increments by 1 each time the 'label' value changes
    data_copy['group'] = (data['label'].diff() != 0).cumsum()

    # Filter the data to select only rows where the 'label' column has a value of 1
    data_anomalies = data_copy[data_copy["label"] == 1]

    # Group by the 'group' column and get the first and last index of each group
    grouped = data_anomalies.groupby('group')
    consecutive_labels_indexes = [(group.index.min(), group.index.max()) for _, group in grouped]

    # Trim the start and end of the anomalies to remove the onset and the offset
    trimmed_anomalies_indexes = []
    anomaly_lengths = []
    for start, end in consecutive_labels_indexes:
        anomaly_length = end - start
        anomaly_lengths.append(anomaly_length)
        trim_amount = int(anomaly_length * trim_percentage / 100)
        trimmed_start = start + trim_amount
        trimmed_end = end - trim_amount
        trimmed_anomalies_indexes.append((trimmed_start, trimmed_end))
    
    # Extract the data
    anomaly_data = []
    for start, end in trimmed_anomalies_indexes:
        subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
        anomaly_data.append(subset_rows)
    
    # Group the data in windows before saving
    anomaly_data = windower(anomaly_data, window_size, stride, num_variables)
    
    anomaly_data = [anomaly_data] + [anomaly_lengths]

    # Save anomaly_data to disk as pickle object
    with open('pickels/anomaly_data_0.pkl', 'wb') as file:
        pickle.dump(anomaly_data, file)
    
    return trimmed_anomalies_indexes

trim_precentage = 0
window_size = 32
stride = 1
num_variables = 6

anomalies_indexes = anomalies(901, trim_precentage, window_size, stride, num_variables)

# Read the windowed anomalous data
file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

window_size_high = 32
window_size_med = 16
window_size_low = 8

X = anomalies_windows[0]
lengths = anomalies_windows[-1][:-1]
lengths = [i - window_size_high + 1 for i in lengths]

index = 14 # This index would give an error with the old indexing set up
print(X[0][index])
# print(X[1][index:index+17][0])
# print(X[1][index:index+17][-1])
print(dater(901, X[0][index]))
# print(dater(901, X[1][index:index+17]))
