import pickle
import numpy as np

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
anomaly_indices = np.where(y == 1)[0][:100] # Selecting the first 100 anomalies in this case

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

# Select the anomaly to explain
anomaly_to_explain = 0

# Retrieve the decision paths of this anomaly across all trees in the RF
decision_paths = all_decision_paths[0]

# Get the indices where the anomaly has passed through
passed_nodes_indices = [np.where(decision_path == 1)[1] for decision_path in decision_paths]

# Access the decision tree from the Random Forest model
# tree = model.estimators_[0]  # Choose the index of the tree you're interested in

# Get the thresholds and feature values of the nodes in each decision tree in the Random Forest
tree_feature_thresholds = [model.estimators_[i].tree_.threshold[e] for i, e in enumerate(passed_nodes_indices)]
tree_feature_indices = [model.estimators_[i].tree_.feature[e] for i, e in enumerate(passed_nodes_indices)]

# Define feature names (for a med tree in this case)
feature_names = ['am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                'am+0', 'co+0', 'do+0', 'ph+0', 'wt+0', 'tu+0',
                'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7']

# Continue here. The wt+7 has to be removed at the end of each element because it corresponds to the -2 index of the leaves
subset_feature_names = []
for i in tree_feature_indices:
    subset_feature_names.append([feature_names[j] for j in i[:-1].tolist()])

print(subset_feature_names)