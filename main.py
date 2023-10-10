import pickle
import numpy as np

# Read the data
file_anomalies = open('anomaly_data.pkl', 'rb')
anomalies = pickle.load(file_anomalies)
file_anomalies.close()

file_background = open('background_data.pkl', 'rb')
background = pickle.load(file_background)
file_background.close()

def windower(data, num_variables, window_size, stride):

    """Takes a 2D array with multivariate time series data
    and creates sliding windows. The arrays store the different
    variables in a consecutive manner. E.g. [first 6 variables,
    next 6 variables, and so on]."""
    
    windows = []
    for i in data:
        
        # Get the number of windows
        num_windows = (len(i) - window_size * num_variables) // num_variables + 1
        
        # Create windows
        for j in range(0, num_windows, stride * num_variables):
            window = i[j:j+window_size * num_variables]  # Extract a window of 4 time steps (4 * 6 variables)
            windows.append(window)

    # Convert the result to a NumPy array
    windows = np.array(windows)
    
    return windows

anomalies_windows = windower(anomalies, num_variables=6, window_size=4, stride=1)
background_windows = windower(background, num_variables=6, window_size=4, stride=1)

# Generate labels for each window
anomalies_labels = np.array([1 for i in anomalies_windows])
background_labels = np.array([0 for i in background_windows])

# Concatenate arrays
X = np.concatenate((anomalies_windows, background_windows))
y = np.concatenate((anomalies_labels, background_labels))

# Shuffle data
combined = np.column_stack((X, y))
np.random.seed(0)
np.random.shuffle(combined)

# Split the shuffled array back into data and labels
X, y = combined[:, :-1], combined[:, -1]

# Train a classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)

# Split the shuffled data into the training and testing set
X_train, y_train = X[:int(len(X) * 0.75)], y[:int(len(X) * 0.75)]
X_test, y_test = X[int(len(X) * 0.75):], y[int(len(X) * 0.75):]

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the model to disk
filename = 'models/rf_model.sav'
pickle.dump(model, open(filename, 'wb'))

from sklearn.metrics import confusion_matrix as cm
confusion_matrix = cm(y_test, model.predict(X_test))
print(confusion_matrix)

# Get the number of rows labeled as anomalies in y_test
num_anomalies = len([i for i in y_test if i==1])
print('Number of anomalies', num_anomalies)
