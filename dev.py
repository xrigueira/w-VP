import pickle
import numpy as np
import matplotlib.pyplot as plt

from main import windower

# Read the data
file_anomalies = open('anomaly_data.pkl', 'rb')
anomalies = pickle.load(file_anomalies)
file_anomalies.close()

file_background = open('background_data.pkl', 'rb')
init_background = pickle.load(file_background)
file_background.close()

file_background = open('background_data_2.pkl', 'rb')
background = pickle.load(file_background)
file_background.close()

# Make windows
anomalies_windows = windower(anomalies, num_variables=6, window_size=4, stride=1)
init_background_windows = windower(init_background, num_variables=6, window_size=4, stride=1)
background_windows = windower(background, num_variables=6, window_size=4, stride=1)

# Variable name change to follow best practives in ML
X = background_windows

# Shuffle the data
np.random.seed(0)
np.random.shuffle(X)

# Load the model
filename = 'models/rf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Get the results from each tree
trees = loaded_model.estimators_

tree_classifications = [tree.predict(X) for tree in trees]

# Get the average score for each windows
score_Xs = np.mean(tree_classifications, axis=0)

# plt.plot(score_Xs)

# Get the indexes of those windows that are anomalies and background in the new data
indexes_anomalies_windows = list(np.where(score_Xs >= 0.75)[0])
indexes_background_windows = list(np.where(score_Xs <= 0.20)[0])

# Extract those new anomaly and background windows
add_anomalies_windows = background_windows[indexes_anomalies_windows]
add_background_windows = background_windows[indexes_background_windows]

# Conactenate new data with old data
anomalies_windows = np.vstack((anomalies_windows, add_anomalies_windows))
background_windows = np.vstack((background_windows, add_background_windows))

# Retrain the model with the updated anomaly and background data

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

# Load the model
filename = 'models/rf_model.sav'
model = pickle.load(open(filename, 'rb'))

# Increase estimators and set warm_start to True
model.n_estimators += 10
model.warm_start = True

# Split the shuffled data into the training and testing set
X_train, y_train = X[:int(len(X) * 0.75)], y[:int(len(X) * 0.75)]
X_test, y_test = X[int(len(X) * 0.75):], y[int(len(X) * 0.75):]

# Fit the model to the training data
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix as cm
confusion_matrix = cm(y_test, model.predict(X_test))
print(confusion_matrix)

# Get the number of rows labeled as anomalies in y_test
num_anomalies = len([i for i in y_test if i==1])
print('Number of anomalies', num_anomalies)

# Save the model to disk
filename = 'models/rf_model_1.sav'
pickle.dump(model, open(filename, 'wb'))

# I have to put carefully out everyhing together in a Class (anomalies_background, main, and test)
