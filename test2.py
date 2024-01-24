import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

resolution = 'high'

window_size_high, window_size_med, window_size_low = 32, 16, 8

iteration = 7

# Load model
filename = f'models/rf_model_high_{iteration}.sav'
model = pickle.load(open(filename, 'rb'))

# Load the anomalies data
file_anomalies = open(f'pickels/anomaly_data_test.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Get windowed data and rename it to X
X = anomalies_windows[0]

lengths = anomalies_windows[-1]
number_windows_high = [i - window_size_high + 1 for i in lengths]
number_windows_med = [i - window_size_med + 1 for i in lengths]
number_windows_low = [i - window_size_low + 1 for i in lengths] 

# Replicate the explainer_test function

# Create an empty list to store decision path for each window and all of the decision paths
decision_paths = []

# window = 0
data = X[0][0]
for tree in model.estimators_:
    tree_decision_paths = tree.decision_path(data[np.newaxis, :]).toarray()
    decision_paths.append(tree_decision_paths)

# Get the indices where the window has passed through
passed_nodes_indices = [np.where(decision_path == 1)[1] for decision_path in decision_paths]

# Get the thresholds and feature values of the nodes in each decision tree in the Random Forest
tree_feature_thresholds = [model.estimators_[i].tree_.threshold[e] for i, e in enumerate(passed_nodes_indices)]
tree_feature_indices = [model.estimators_[i].tree_.feature[e] for i, e in enumerate(passed_nodes_indices)]

# Define feature names for all resolution levels
feature_names_high = [
                'am-16', 'co-16', 'do-16', 'ph-16', 'wt-16', 'tu-16',
                'am-15', 'co-15', 'do-15', 'ph-15', 'wt-15', 'tu-15',
                'am-14', 'co-14', 'do-14', 'ph-14', 'wt-14', 'tu-14',
                'am-13', 'co-13', 'do-13', 'ph-13', 'wt-13', 'tu-13',
                'am-12', 'co-12', 'do-12', 'ph-12', 'wt-12', 'tu-12',
                'am-11', 'co-11', 'do-11', 'ph-11', 'wt-11', 'tu-11',
                'am-10', 'co-10', 'do-10', 'ph-10', 'wt-10', 'tu-10',
                'am-9', 'co-9', 'do-9', 'ph-9', 'wt-9', 'tu-9',
                'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8',
                'am+9', 'co+9', 'do+9', 'ph+9', 'wt+9', 'tu+9',
                'am+10', 'co+10', 'do+10', 'ph+10', 'wt+10', 'tu+10',
                'am+11', 'co+11', 'do+11', 'ph+11', 'wt+11', 'tu+11',
                'am+12', 'co+12', 'do+12', 'ph+12', 'wt+12', 'tu+12',
                'am+13', 'co+13', 'do+13', 'ph+13', 'wt+13', 'tu+13',
                'am+14', 'co+14', 'do+14', 'ph+14', 'wt+14', 'tu+14',
                'am+15', 'co+15', 'do+15', 'ph+15', 'wt+15', 'tu+15',
                'am+16', 'co+16', 'do+16', 'ph+16', 'wt+16', 'tu+16'
                ]

feature_names_med = [
                'am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7',
                'am+8', 'co+8', 'do+8', 'ph+8', 'wt+8', 'tu+8'
                ]

feature_names_low = [
                'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4'
                ]

# Select the resolution of the feature names
feature_names = feature_names_high if resolution == 'high' else feature_names_med if resolution == 'med' else feature_names_low

# The wt+7 has to be removed at the end of each element because it corresponds to the -2 index of the leaves
subset_feature_names = []
for i in tree_feature_indices:
    subset_feature_names.append([feature_names[j] for j in i[:-1].tolist()])

subset_feature_thresholds = []
for i in tree_feature_thresholds:
    subset_feature_thresholds.append(i[:-1].tolist())

# Variable-position plot
# Extract variable names and their positions
variables = {}
for sublist in subset_feature_names:
    for i, item in enumerate(sublist):
        var = item.split('-')[0].split('+')[0]
        if var not in variables:
            variables[var] = []
        variables[var].append(i)

# Create a 2D array for the heatmap
max_len = max([len(sublist) for sublist in subset_feature_names])
heatmap_data = np.zeros((len(variables), max_len))

for i, var in enumerate(variables.keys()):
    for pos in variables[var]:
        heatmap_data[i, pos] += 1

# Create the heatmap
heatmap_data = heatmap_data.astype(int) # Convert data to int

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, xticklabels=range(max_len), yticklabels=list(variables.keys()), cmap='viridis', annot=True, fmt="d")
plt.xlabel('Position')
plt.ylabel('Variable')
plt.title(f'Variable importance window {0}')
plt.show()

# plt.savefig(f'images/position_{window_to_explain}.png')

# Variable-threshold plot
# Read the data and get the mean for each variable
station = 901
data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])

stats_dict = {}
var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
for i, e in enumerate(data.iloc[:, 1:7]):
    stats_dict[var_names[i]] = data[e].quantile(0.25), data[e].mean(), data[e].quantile(0.75)

variables_dict = {}
for sublist, sublist_thresholds in zip(subset_feature_names, subset_feature_thresholds):
    for var, threshold in zip(sublist, sublist_thresholds):
        var_type = var.split('-')[0].split('+')[0]  # Extract variable type
        if var_type not in variables_dict:
            variables_dict[var_type] = {}
        if var not in variables_dict[var_type]:
            variables_dict[var_type][var] = []
        variables_dict[var_type][var].append(threshold)

# Add the missing feature names to the dictionary
for feature_name in feature_names:
    # Extract variable type and specific feature
    var_type = feature_name.split('-')[0].split('+')[0]
    specific_feature = feature_name

    # If the variable type exists in the dictionary
    if var_type in variables_dict:
        # If the specific feature doesn't exist in the sub-dictionary
        if specific_feature not in variables_dict[var_type]:
            # Add it with an empty list as the value
            variables_dict[var_type][specific_feature] = []
    else:
        # If the variable type doesn't exist in the dictionary, add it
        variables_dict[var_type] = {specific_feature: []}

# Sort the dictionary by keys
sorted_dict = {}
for var_type, var_dict in variables_dict.items():
    # Extract the numeric part, convert to integer, keep the sign and sort
    sorted_keys = sorted(var_dict.keys(), key=lambda x: int(x.split(var_type)[1]) if var_type in x else 0)
    sorted_dict[var_type] = {key: var_dict[key] for key in sorted_keys}

# Calculate the distance between the means
distance_dict = {}
for var_type, var_dict in sorted_dict.items():
    if var_type not in distance_dict:
        distance_dict[var_type] = {}
    for key, values in var_dict.items():
            mean_distance = stats_dict[var_type][1] - np.mean(values) # Here is where I can choose: 0: q1, 1: mean, 2: q3
            distance_dict[var_type][key] = mean_distance

# Plot the heatmap
# Turn into a numpy array
distances = [np.array(list(distance_dict[var].values())) for var in var_names]
heatmap_data = np.vstack(distances)

# Define xticklabels
xticklabels_high = [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 
                    +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16]
xticklabels_med = [-8, -7, -6, -5, -4, -3, -2, -1, 
                    +1, +2, +3, +4, +5, +6, +7, +8]
xticklabels_low = [-4, -3, -2, -1, 
                    +1, +2, +3, +4]

# Select the resolution of the feature names
xticklabels = xticklabels_high if resolution == 'high' else xticklabels_med if resolution == 'med' else xticklabels_low

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, xticklabels=xticklabels, yticklabels=var_names, cmap='jet', vmin=-0.8, vmax=0.8)
plt.xlabel('Position')
plt.ylabel('Variable')
plt.title(f'Threshold distance to the mean {0}')
plt.show()

# plt.savefig(f'images/thresholds_{window_to_explain}.png')
