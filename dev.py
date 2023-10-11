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
anomaly_windows = windower(anomalies, num_variables=6, window_size=4, stride=1)
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
indexes_anomaly_windows = list(np.where(score_Xs >= 0.75)[0])
indexes_background_windows = list(np.where(score_Xs <= 0.20)[0])

# Extract those new anomaly and background windows
add_anomaly_windows = background_windows[indexes_anomaly_windows]
add_background_windows = background_windows[indexes_background_windows]

# Conactenate new data with old data
anomaly_windows = np.vstack((anomaly_windows, add_anomaly_windows))
background_windows = np.vstack((background_windows, add_background_windows))

# Now I have to train the update the model using the new anomaly and background windows, then
# save it and repeate the process. I have to put carefully out everyhing together in a Class (anomalies_background, main, and test)
# OJO I have to define a conditional for then indexes_anomaly_windows or background are empty. In this case it would just pass
# and move to the next iterartion. This the case of the data in background_data_1. 
