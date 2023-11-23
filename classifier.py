import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.cluster import KMeans

# Set the maximum number of CPU cores for joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set this to the number of cores you want to use

# Define function to convert binary decision paths to integers
def convert2int(binary_array):
    # Initialize an empty list to store the result
    result = []

    # Initialize counters for consecutive ones and zeros
    count_ones = 0
    count_zeros = 0

    # Iterate through the elements of the binary array
    for bit in binary_array:
        if bit == 1:
            # If the current digit is 1, increment the count for consecutive ones
            count_ones += 1

            # If there were consecutive zeros before, append the count to the result
            if count_zeros > 0:
                result.append(count_zeros)

            # Reset the count for consecutive zeros
            count_zeros = 0
        else:
            # If the current digit is 0, increment the count for consecutive zeros
            count_zeros += 1

            # If there were consecutive ones before, append the count to the result
            if count_ones > 0:
                result.append(count_ones)

            # Reset the count for consecutive ones
            count_ones = 0

    # Append the final count if there are consecutive ones or zeros at the end
    if count_ones > 0:
        result.append(count_ones)
    elif count_zeros > 0:
        result.append(count_zeros)
    
    # Convert the result array to an integer
    result = int(''.join(map(str, result)))

    return result

# Load a model. I am using the last model in this case -- 9.
iteration = 9
filename = f'models/rf_model_med_{iteration}.sav'
model = pickle.load(open(filename, 'rb'))

# Load the data
file_anomalies = open(f'pickels/anomaly_data_{iteration}.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the previous windows background
file_background = open(f'pickels/background_data_{iteration}.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close

# Define the labels
anomalies_labels = []
for i in range(len(anomalies_windows)):
    anomalies_labels.append(np.array([1 for j in anomalies_windows[i]]))

background_labels = []
for i in range(len(background_windows)):
    background_labels.append(np.array([0 for j in background_windows[i]]))

# Concatenate array
X = []
for i in range(len(anomalies_windows)):
    X.append(np.concatenate((anomalies_windows[i], background_windows[i])))

y = []
for i in range(len(anomalies_windows)):
    y.append(np.concatenate((anomalies_labels[i], background_labels[i])))

# Get the data corresponding to the resolution of the model (high:0, med:1, low:2)
X, y = X[1], y[1]

# Get the anomaly indices
anomaly_indices = np.where(y == 1)[0][:200] # Selecting the first 1000 anomalies in this case

# Create an empty list to store decision path for each anomaly and all of the decision paths
all_decision_paths = []
anomaly_decision_paths =[]

# Traverse each tree in the Random Forest to get the decision path across all tree for each anomaly
for anomaly in anomaly_indices:
    for tree in model.estimators_:
        tree_decision_path = tree.decision_path(X[anomaly][np.newaxis, :]).toarray()
        anomaly_decision_paths.append(tree_decision_path)
    all_decision_paths.append(anomaly_decision_paths) # The first elemnt would be all decision paths of the first anomaly
    anomaly_decision_paths = []

# Concatenate the decision paths along the columns
decision_paths = [np.concatenate(i, axis=1) for i in all_decision_paths]

# Convert binary paths to integer values
integer_paths = [convert2int(i.flatten()) for i in decision_paths]

# Normalize integer paths
min_integer, max_integer = min(integer_paths), max(integer_paths)
normed_integer_paths = [((i - min_integer)/(max_integer - min_integer)) for i in integer_paths]

# Print number of types
print('Number of unique anomalies:', len(np.unique(integer_paths)))
print('Number of unique normed anomalies:', len(np.unique(normed_integer_paths)))

# Get sine and cosine of each value
sin_cos_integer_paths = np.array([[math.sin(i), math.cos(i)] for i in normed_integer_paths])

# Extract x and y coordinates
x_values = sin_cos_integer_paths[:, 0]
y_values = sin_cos_integer_paths[:, 1]

# Apply K-Means clustering
num_clusters = 3  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=0)
cluster_assignments = kmeans.fit_predict(sin_cos_integer_paths)

# Plot the points with different colors for each cluster
plt.scatter(x_values, y_values, c=cluster_assignments, cmap='turbo', marker='o', label='Points')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('KMeans Clustering Results')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# Print the cluster assignments for each anomaly instance
for i, cluster in enumerate(cluster_assignments):
    print(f"Anomaly instance {anomaly_indices[i]} belongs to cluster {cluster}")
