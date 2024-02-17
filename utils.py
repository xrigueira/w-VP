import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dater(station, window):

    """This function returns the dates corresponding to a window.
    ---------
    Arguments:
    station: The station number.
    window: The window to be converted.
    
    Returns:
    date_indices: The dates corresponding to the window."""

    # Read data
    data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'], index_col=['date'])
    data = data.iloc[:, :-2]

    # Reshape window and define mask
    window = np.array(window).reshape(-1, 6)
    mask = np.zeros(len(data), dtype=bool)
    for window_row in window:
        mask |= (data.values == window_row).all(axis=1)
    
    # Extract dates
    indices = np.where(mask)[0]
    
    date_indices = data.index[indices]
    
    return date_indices

def plotter(data, num_variables, legend, name):
    
    """This function plots the data passed as a 
    numpy arrayoriginal data, for a given resolution 
    level.
    ---------
    Arguments:
    data: The data to be plotted.
    num_variables: The number of variables in the data.
    windowed: Whether the data is windowed or not.
    
    Returns:
    None"""

    variables_names = ["am", "co", "do", "ph", "tu", "wt"]

    data_reshaped = data.reshape(-1, num_variables)
    
    # Plot each variable
    for i in range(num_variables):
        plt.plot(dater(901, data), data_reshaped[:, i], label=f'{variables_names[i]}')

    plt.xticks(rotation=30)
    plt.xlabel('Time/Index')
    plt.ylabel('Variable Value')
    # plt.title(f'Event {name}', loc='left')
    if legend: plt.legend()
    plt.tight_layout()
    # plt.show()

    # Save figure
    plt.savefig(f'images/{name}.png')

    # Close figure
    plt.close()

def explainer(data, model, resolution, name):

    """This function explains the decision of a Random Forest model
    for a given window.
    ---------
    Arguments:
    data: The data to be explained.
    model: The Random Forest model to be explained.
    resolution: The resolution of the model.
    name: The title of the plot.
    
    Returns:
    None."""

    # Create an empty list to store decision path for each window and all of the decision paths
    decision_paths = []

    # Traverse each tree in the Random Forest to get the decision path across all trees for the given window (data)
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
                    'am-16', 'co-16', 'do-16', 'ph-16', 'tu-16', 'wt-16',
                    'am-15', 'co-15', 'do-15', 'ph-15', 'tu-15', 'wt-15',
                    'am-14', 'co-14', 'do-14', 'ph-14', 'tu-14', 'wt-14',
                    'am-13', 'co-13', 'do-13', 'ph-13', 'tu-13', 'wt-13',
                    'am-12', 'co-12', 'do-12', 'ph-12', 'tu-12', 'wt-12',
                    'am-11', 'co-11', 'do-11', 'ph-11', 'tu-11', 'wt-11',
                    'am-10', 'co-10', 'do-10', 'ph-10', 'tu-10', 'wt-10',
                    'am-9', 'co-9', 'do-9', 'ph-9', 'tu-9', 'wt-9',
                    'am-8', 'co-8', 'do-8', 'ph-8', 'tu-8', 'wt-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'tu-7', 'wt-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'tu-6', 'wt-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'tu-5', 'wt-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'tu+5', 'wt+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'tu+6', 'wt+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'tu+7', 'wt+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'tu+8', 'wt+8',
                    'am+9', 'co+9', 'do+9', 'ph+9', 'tu+9', 'wt+9',
                    'am+10', 'co+10', 'do+10', 'ph+10', 'tu+10', 'wt+10',
                    'am+11', 'co+11', 'do+11', 'ph+11', 'tu+11', 'wt+11',
                    'am+12', 'co+12', 'do+12', 'ph+12', 'tu+12', 'wt+12',
                    'am+13', 'co+13', 'do+13', 'ph+13', 'tu+13', 'wt+13',
                    'am+14', 'co+14', 'do+14', 'ph+14', 'tu+14', 'wt+14',
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt15',
                    'am+16', 'co+16', 'do+16', 'ph+16', 'tu+16', 'wt+16'
                    ]

    feature_names_med = [
                    'am-8', 'co-8', 'do-8', 'ph-8', 'tu-8', 'wt-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'tu-7', 'wt-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'tu-6', 'wt-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'tu-5', 'wt-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'tu+5', 'wt+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'tu+6', 'wt+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'tu+7', 'wt+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'tu+8', 'wt+8',
                    ]

    feature_names_low = [
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
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
    # plt.show()

    plt.savefig(f'images/{name}_var.png')

    # Close figure
    plt.close()

    # Variable-threshold plot
    # Read the data and get the mean for each variable
    station = 901
    df = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    
    stats_dict = {}
    var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
    for i, e in enumerate(df.iloc[:, 1:7]):
        stats_dict[var_names[i]] = df[e].quantile(0.25), df[e].mean(), df[e].quantile(0.75)
    
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
    # plt.show()

    plt.savefig(f'images/{name}_thre.png')

    # Close figure
    plt.close()

# Mean-position plot
def mean_plotter(data, resolution, num_variables, name):
    
    # Read the data and get the mean for each variable
    station = 901
    df = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])

    stats_dict = {}
    var_names = ['am', 'co', 'do', 'ph', 'tu', 'wt']
    for i, e in enumerate(df.iloc[:, 1:7]):
        stats_dict[var_names[i]] = df[e].quantile(0.25), df[e].mean(), df[e].quantile(0.75)

    # Reshape the data
    data = data.reshape(-1, num_variables).T

    # Get the difference between the variables at each position and the mean
    for i, arr in enumerate(data):
        # Get the corresponding variable name
        var_name = var_names[i]
        
        # Get the mean for this variable
        mean = stats_dict[var_name][1]
        
        # Subtract the mean from each element in the array
        data[i] = arr - mean
    
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
    sns.heatmap(data, xticklabels=xticklabels, yticklabels=var_names, cmap='jet', vmin=-0.8, vmax=0.8)
    plt.xlabel('Position')
    plt.ylabel('Variable')
    # plt.show()

    plt.savefig(f'images/{name}_mean.png')

    # Close figure
    plt.close()

def tree_plotter(model, resolution):

    """This function plots a tree of a Random Forest model.
    ---------
    Arguments:
    model: The Random Forest model to plot.
    resolution: The resolution of the model.

    Returns:
    None.
    """

    # Plot an estimator (tree) of the Random Forest model
    from sklearn.tree import export_graphviz

    # Define feature names for all resolution levels
    feature_names_high = [
                    'am-16', 'co-16', 'do-16', 'ph-16', 'tu-16', 'wt-16',
                    'am-15', 'co-15', 'do-15', 'ph-15', 'tu-15', 'wt-15',
                    'am-14', 'co-14', 'do-14', 'ph-14', 'tu-14', 'wt-14',
                    'am-13', 'co-13', 'do-13', 'ph-13', 'tu-13', 'wt-13',
                    'am-12', 'co-12', 'do-12', 'ph-12', 'tu-12', 'wt-12',
                    'am-11', 'co-11', 'do-11', 'ph-11', 'tu-11', 'wt-11',
                    'am-10', 'co-10', 'do-10', 'ph-10', 'tu-10', 'wt-10',
                    'am-9', 'co-9', 'do-9', 'ph-9', 'tu-9', 'wt-9',
                    'am-8', 'co-8', 'do-8', 'ph-8', 'tu-8', 'wt-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'tu-7', 'wt-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'tu-6', 'wt-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'tu-5', 'wt-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'tu+5', 'wt+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'tu+6', 'wt+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'tu+7', 'wt+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'tu+8', 'wt+8',
                    'am+9', 'co+9', 'do+9', 'ph+9', 'tu+9', 'wt+9',
                    'am+10', 'co+10', 'do+10', 'ph+10', 'tu+10', 'wt+10',
                    'am+11', 'co+11', 'do+11', 'ph+11', 'tu+11', 'wt+11',
                    'am+12', 'co+12', 'do+12', 'ph+12', 'tu+12', 'wt+12',
                    'am+13', 'co+13', 'do+13', 'ph+13', 'tu+13', 'wt+13',
                    'am+14', 'co+14', 'do+14', 'ph+14', 'tu+14', 'wt+14',
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt15',
                    'am+16', 'co+16', 'do+16', 'ph+16', 'tu+16', 'wt+16'
                    ]

    feature_names_med = [
                    'am-8', 'co-8', 'do-8', 'ph-8', 'tu-8', 'wt-8',
                    'am-7', 'co-7', 'do-7', 'ph-7', 'tu-7', 'wt-7',
                    'am-6', 'co-6', 'do-6', 'ph-6', 'tu-6', 'wt-6',
                    'am-5', 'co-5', 'do-5', 'ph-5', 'tu-5', 'wt-5',
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    'am+5', 'co+5', 'do+5', 'ph+5', 'tu+5', 'wt+5',
                    'am+6', 'co+6', 'do+6', 'ph+6', 'tu+6', 'wt+6',
                    'am+7', 'co+7', 'do+7', 'ph+7', 'tu+7', 'wt+7',
                    'am+8', 'co+8', 'do+8', 'ph+8', 'tu+8', 'wt+8',
                    ]

    feature_names_low = [
                    'am-4', 'co-4', 'do-4', 'ph-4', 'tu-4', 'wt-4',
                    'am-3', 'co-3', 'do-3', 'ph-3', 'tu-3', 'wt-3',
                    'am-2', 'co-2', 'do-2', 'ph-2', 'tu-2', 'wt-2',
                    'am-1', 'co-1', 'do-1', 'ph-1', 'tu-1', 'wt-1',
                    'am+1', 'co+1', 'do+1', 'ph+1', 'tu+1', 'wt+1',
                    'am+2', 'co+2', 'do+2', 'ph+2', 'tu+2', 'wt+2',
                    'am+3', 'co+3', 'do+3', 'ph+3', 'tu+3', 'wt+3',
                    'am+4', 'co+4', 'do+4', 'ph+4', 'tu+4', 'wt+4',
                    ]

    # Select the resolution of the feature names
    feature_names = feature_names_high if resolution == 'high' else feature_names_med if resolution == 'med' else feature_names_low

    # Plot the tree ([0] for the first tree in the RF)
    export_graphviz(model.estimators_[0], out_file='tree.dot',
                    feature_names=feature_names, # Number of data points in a window
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_0.png', '-Gdpi=600'])

def summarizer(data, num_variables):
    
    """Gets a 1D data window and returns a summarized version 
    with the mean for each variable.
    ----------
    Arguments:
    data (np.array): The data to be summarized.
    num_variables (int): The number of variables in the data.
    
    Returns:
    data_summarized (np.array): The summarized data."""

    data_summarized = []
    for i in data:
        i = i.reshape(-1, num_variables)
        data_summarized.append(i.mean(axis=0))
    
    return np.array(data_summarized)
