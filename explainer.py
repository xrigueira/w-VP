import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data_type = 'anomalies' # 'anomalies' or 'background
resolution = 'high' # 'high', 'med', 'low'

if resolution == 'high':
    window_size = 32
elif resolution == 'med':
    window_size = 16
elif resolution == 'low':
    window_size = 8

# Load a model. I am using the last model in this case -- 9.
iteration = 7
filename = f'models/rf_model_{resolution}_{iteration}.sav'
model = pickle.load(open(filename, 'rb'))


if data_type == 'anomalies':
    
    # Load the anomalies data
    file_anomalies = open(f'pickels/anomaly_data_test.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Get the data corresponding to the resolution of the model and rename it to X
    if resolution == 'high':
        X = anomalies_windows[0][0]
    elif resolution == 'med':
        X = anomalies_windows[0][1]
    elif resolution == 'low':
        X = anomalies_windows[0][2]
    
    lengths = anomalies_windows[-1]
    number_windows = [i - window_size + 1 for i in lengths]

elif data_type == 'background':
    
    # Load the background data
    file_background = open(f'pickels/background_data_test.pkl', 'rb')
    background_windows = pickle.load(file_background)
    file_background.close

    # Get the data corresponding to the resolution of the model and rename it to X
    if resolution == 'high':
        X = background_windows[0][0]
    elif resolution == 'med':
        X = background_windows[0][1]
    elif resolution == 'low':
        X = background_windows[0][2]
    
    lengths = background_windows[-1]
    number_windows = [i - window_size + 1 for i in lengths]

X = X[:20] # Subset the data to the first 20 anomalies just for testing purposes

# Create an empty list to store decision path for each anomaly and all of the decision paths
all_decision_paths = []
anomaly_decision_paths =[]

# Traverse each t
# ree in the Random Forest to get the decision path across all tree for each anomaly
for anomaly in range(len(X)):
    for tree in model.estimators_:
        tree_decision_path = tree.decision_path(X[anomaly][np.newaxis, :]).toarray()
        anomaly_decision_paths.append(tree_decision_path)
    all_decision_paths.append(anomaly_decision_paths) # The first elemnt would be all decision paths of the first anomaly
    anomaly_decision_paths = []

# Select the anomaly to explain
anomaly_to_explain = 0

# Retrieve the decision paths of this anomaly across all trees in the RF
decision_paths = all_decision_paths[anomaly_to_explain]

# Get the indices where the anomaly has passed through
passed_nodes_indices = [np.where(decision_path == 1)[1] for decision_path in decision_paths]

# Access the decision tree from the Random Forest model
# tree = model.estimators_[0]  # Choose the index of the tree you're interested in

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
plt.title(f'Variable importance anomaly {anomaly_to_explain}')
plt.show()

# Variable-threshold plot
variables_dict = {}
for sublist, sublist_thresholds in zip(subset_feature_names, subset_feature_thresholds):
    for var, threshold in zip(sublist, sublist_thresholds):
        var_type = var.split('-')[0].split('+')[0]  # Extract variable type
        if var_type not in variables_dict:
            variables_dict[var_type] = {}
        if var not in variables_dict[var_type]:
            variables_dict[var_type][var] = []
        variables_dict[var_type][var].append(threshold)

# Define a color mapping for the variable types
color_mapping = {
    'am': '#ff6961',
    'co': '#ffb347',
    'do': '#aec6cf',
    'ph': '#b39eb5',
    'tu': '#fdfd96',
    'wt': '#77dd77'
}

# Create a violin plot for each variable type
for var_type, var_dict in variables_dict.items():
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([(key, var) for key, values in var_dict.items() for var in values], columns=['Variable', 'Threshold'])
    
    # Extract the numeric part of 'Variable' for sorting
    df['SortKey'] = df['Variable'].apply(lambda x: int(x.split(var_type)[1]) if var_type in x else 0)

    # Sort the DataFrame by 'SortKey'
    df = df.sort_values('SortKey', ascending=False)

    # Drop the 'SortKey' as it's no longer needed
    df = df.drop('SortKey', axis=1)

    # Create the violin plot with the color for the current variable type
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='Variable', y='Threshold', data=df, order=df['Variable'].unique(), color=color_mapping.get(var_type, 'black'))
    plt.title(f'Violin plot for {var_type} variables')
    plt.show()
