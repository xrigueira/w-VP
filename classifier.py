import pickle
import numpy as np

from sklearn.cluster import KMeans

# Random Forest docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Decision Tree Classifier (estimator_) docs: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# This is key: Decision Tree Structure docs: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

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
anomaly_indices = np.where(y == 1)[0]

# Create an empty list to store decision paths
decision_paths = []

# Traverse each tree in the Random Forest
for tree in model.estimators_:
    # Get the decision path for each anomaly instance
    tree_decision_paths = tree.decision_path(X[anomaly_indices]).toarray()
    decision_paths.append(tree_decision_paths)

# Concatenate the decision paths along the columns
all_decision_paths = np.concatenate(decision_paths, axis=1)

# Convert the binary decision paths to integers
integer_decision_paths = all_decision_paths.dot(1 << np.arange(all_decision_paths.shape[-1] - 1, -1, -1))

# Reshape the integer decision paths to have one row per anomaly instance
integer_decision_paths = integer_decision_paths.reshape(len(anomaly_indices), -1)

# Apply K-Means clustering
num_clusters = 3  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(integer_decision_paths)

# Print the cluster assignments for each anomaly instance
for i, cluster in enumerate(cluster_assignments):
    print(f"Anomaly instance {anomaly_indices[i]} belongs to cluster {cluster}")