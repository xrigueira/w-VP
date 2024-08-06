import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'
from matplotlib.dates import DateFormatter

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

def window_plotter(data, num_variables, legend, event_number, station, type):
    
    """This function plots the data window passed as a 
    numpy array original data. Hence, the resolution is
    given by the window data.
    ---------
    Arguments:
    data: The data window to be plotted.
    num_variables: The number of variables in the data.
    station: the station number.
    legend: Whether to show the legend or not.
    name: The title of the plot.
    
    Returns:
    None"""

    variables_names = ["am", "co", "do", "ph", "tu", "wt"]

    data_reshaped = data.reshape(-1, num_variables)
    
    # Plot each variable
    fig, ax = plt.subplots(figsize=(14, 10))  # Set the size of the plot
    for i in range(num_variables):
        x = dater(station, data)
        # if len(x) != 32: x = range(32)
        ax.plot(x, data_reshaped[:, i], label=f'{variables_names[i]}', linewidth=4)
    
    # Set the date format for the x-axis
    date_format = DateFormatter('%Y-%m-%d %Hh')
    ax.xaxis.set_major_formatter(date_format)

    # Set the y-axis limit to 1
    ax.set_ylim(0, 1)

    # Change the fontsize of the ticks
    ax.tick_params(axis='x', which='both', rotation=30, labelsize=23)
    ax.tick_params(axis='y', which='both', labelsize=23)

    # Define axes limits, title and labels
    ax.set_xlabel('Date', fontsize=27)
    ax.set_ylabel('Normalized values', fontsize=27)
    # ax.set_title(f'Event {event_number}', fontsize=20)
    
    if legend: ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=False, ncol=6, fontsize=23)
    plt.tight_layout()
    # plt.show()

    # Save figure
    plt.savefig(f'results/event_{station}_{type}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

    # Close figure
    plt.close()

def event_plotter(starts_ends, X, event_number, station, type):

    event_start_high = starts_ends[event_number][0][0]
    event_end_high = starts_ends[event_number][0][1]

    event_data = X[0][event_start_high]
    for i in range(event_start_high + 1, event_end_high):
        
        # Get the last row of the anomaly
        last_row = X[0][i][-6:]
        
        # Add the last row to anomaly_data
        event_data = np.concatenate((event_data, last_row), axis=0)

    window_plotter(data=event_data, num_variables=6, legend=True, event_number=event_number, station=station, type=type)

# Depth function
def depths(starts_ends, X, models: list, event_number: int):
    
    """This function extracts the depths at which each instance of the
    feature vectors (window) that make up given eventa appears from the 
    decision paths of a Random Forest model considering all trees across 
    all resolutions.
    ---------
    Arguments:
    starts_ends: The start and end indices of each window at all resolutions (len=3).
    X: The windows data at all resolutions (len=3).
    models: The Random Forest models at all resolutions (len=3).
    event_number: The event number to be explained.

    Returns:
    variables_depth: The variables and their depth in each path across all trees.
    max_depth: The maximum depth of the decision paths.
    """

    # Define the resolution names
    resolutions = ['high', 'med', 'low']

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
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt+15',
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

    # Define the variables dictionary
    variables_depths = {}
    variables_thresholds = {}
    variables_distances = {}

    # Set the keys for the variable depths, thresholds, and distances to the variable names in the feature names high
    for resolution in resolutions:
        variables_depths[resolution] = {}
        variables_thresholds[resolution] = {}
        variables_distances[resolution] = {}
        for feature_name in feature_names_high:
            variables_depths[resolution][feature_name] = []
            variables_thresholds[resolution][feature_name] = []
            variables_distances[resolution][feature_name] = []

    # Set the max depth to 0
    max_depth = 0
    for i, (model, resolution) in enumerate(zip(models, resolutions)):
        
        # Extract the start and end indices for the windows by resolution
        # i = 2   # When working with a single resolution, i = 0 for high, i = 1 for med, and i = 2 for low
        windows = X[i][starts_ends[event_number][i][0]:starts_ends[event_number][i][1]]
        
        # Traverse each window to get the decision paths
        for window in windows:
            
            # Create an empty list to store all of the decision paths for each window
            decision_paths = []

            # Traverse each tree in the Random Forest to get the decision path across all trees for the given window (data)
            for tree in model.estimators_:
                tree_decision_paths = tree.decision_path(window[np.newaxis, :]).toarray()
                decision_paths.append(tree_decision_paths)

            # Get the indices where the window has passed through
            passed_nodes_indices = [np.where(decision_path == 1)[1] for decision_path in decision_paths]

            # Get the thresholds and feature values of the nodes in each decision tree in the Random Forest
            tree_feature_thresholds = [model.estimators_[i].tree_.threshold[e] for i, e in enumerate(passed_nodes_indices)]
            tree_feature_indices = [model.estimators_[i].tree_.feature[e] for i, e in enumerate(passed_nodes_indices)]

            # Get the feature values of the nodes in each decision tree in the Random Forest
            tree_feature_values = [window[i] for i in tree_feature_indices]

            # Select the resolution of the feature names
            feature_names = feature_names_high if resolution == 'high' else feature_names_med if resolution == 'med' else feature_names_low

            # The wt+7 has to be removed at the end of each element because it corresponds to the -2 index of the leaves
            subset_feature_names = []
            for i in tree_feature_indices:
                subset_feature_names.append([feature_names[j] for j in i[:-1].tolist()])

            subset_feature_thresholds = []
            for i in tree_feature_thresholds:
                subset_feature_thresholds.append(i[:-1].tolist())

            subset_feature_values = []
            for i in tree_feature_values:
                subset_feature_values.append(i[:-1].tolist())

            # Determine if the node conditions (variable, threshold, value) are satisfied
            conditions_satisfied = []
            for i, (features, thresholds, values) in enumerate(zip(subset_feature_names, subset_feature_thresholds, subset_feature_values)):
                conditions_satisfied.append([True if values[j] <= thresholds[j] else False for j in range(len(features))])
            
            # # Detailed verbal condition satisfaction printout. I levae it commented out to avoid massive printouts, when dealing with several trees, windows, and resolutions
            # for i, (features, values, thresholds, satisfied) in enumerate(zip(subset_feature_names, subset_feature_values, subset_feature_thresholds, conditions_satisfied)):
            #     print(f"Tree {i}")
            #     for j, (feature, value, threshold, is_satisfied) in enumerate(zip(features, values, thresholds, satisfied)):
            #         print(f"Node {i}-{j}: Feature '{feature}' with value {value} {'<= ' if is_satisfied else '> '}threshold {threshold}")

            # Extract variable names and their positions
            for features in subset_feature_names:
                for i, feature in enumerate(features):
                    if feature not in variables_depths[resolution]:
                        variables_depths[resolution][feature] = []
                    variables_depths[resolution][feature].append(i)

            # Extract the thresholds for each variable 
            for features, thresholds in zip(subset_feature_names, subset_feature_thresholds):
                for feature, threshold in zip(features, thresholds):
                    if feature not in variables_thresholds[resolution]:
                        variables_thresholds[resolution][feature] = []
                    variables_thresholds[resolution][feature].append(threshold)
            
            # Extract the distance between the threshold and the value for each variable
            for features, thresholds, values in zip(subset_feature_names, subset_feature_thresholds, subset_feature_values):
                for feature, threshold, value in zip(features, thresholds, values):
                    if feature not in variables_distances[resolution]:
                        variables_distances[resolution][feature] = []
                    distance = value - threshold
                    variables_distances[resolution][feature].append(distance)
            
            # Calculate the depth and update the value
            depth = max([len(sublist) for sublist in subset_feature_names])
            max_depth = max(max_depth, depth)

    return variables_depths, variables_thresholds, variables_distances, max_depth

def normalizer(dictionary):

    """This function normalizes the values of a dictionary, so they
    can be added together and weighted.
    The feature vector of all resolutions (high, med, low) share
    the items between -4, and 4. Thefore, if not normalized, the
    results will be biased.
    Same with the items between -8 and -5 and 5 and 8, which are
    present in both the high and medium resolutions.
    ---------
    Arguments:
    dictionary: The dictionary to be normalized.

    Returns:
    dictionary: The normalized dictionary.
    """

    # Define the resolutions
    resolutions = ['high', 'med', 'low']
    
    # Normalize the values between 0 and 1 for each resolution separately
    for resolution in resolutions:

        # Initialize the list to store all values
        am_values, co_values, do_values, ph_values, tu_values, wt_values = [], [], [], [], [], []

        # Extract the values for each variable
        for key in dictionary[resolution].keys():
            if key.startswith('am'):
                am_values.extend(dictionary[resolution][key])
            elif key.startswith('co'):
                co_values.extend(dictionary[resolution][key])
            elif key.startswith('do'):
                do_values.extend(dictionary[resolution][key])
            elif key.startswith('ph'):
                ph_values.extend(dictionary[resolution][key])
            elif key.startswith('tu'):
                tu_values.extend(dictionary[resolution][key])
            elif key.startswith('wt'):
                wt_values.extend(dictionary[resolution][key])
        
        # Find the maximum and minimum values for each variable
        max_am, min_am = max(am_values), min(am_values)
        max_co, min_co = max(co_values), min(co_values)
        max_do, min_do = max(do_values), min(do_values)
        max_ph, min_ph = max(ph_values), min(ph_values)
        max_tu, min_tu = max(tu_values), min(tu_values)
        max_wt, min_wt = max(wt_values), min(wt_values)

        # Normalize the values for each variable
        for key in dictionary[resolution].keys():
            if key.startswith('am') and (max_am - min_am) != 0:
                dictionary[resolution][key] = [(x - min_am) / (max_am - min_am) for x in dictionary[resolution][key]]
            elif key.startswith('co') and (max_co - min_co) != 0:
                dictionary[resolution][key] = [(x - min_co) / (max_co - min_co) for x in dictionary[resolution][key]]
            elif key.startswith('do') and (max_do - min_do) != 0:
                dictionary[resolution][key] = [(x - min_do) / (max_do - min_do) for x in dictionary[resolution][key]]
            elif key.startswith('ph') and (max_ph - min_ph) != 0:
                dictionary[resolution][key] = [(x - min_ph) / (max_ph - min_ph) for x in dictionary[resolution][key]]
            elif key.startswith('tu') and (max_tu - min_tu) != 0:
                dictionary[resolution][key] = [(x - min_tu) / (max_tu - min_tu) for x in dictionary[resolution][key]]
            elif key.startswith('wt') and (max_wt - min_wt) != 0:
                dictionary[resolution][key] = [(x - min_wt) / (max_wt - min_wt) for x in dictionary[resolution][key]]
    
    return dictionary

def weighter(multiresolution_map):
    
    """This applies weights to the multiresolution attention map
    in order to normalize the values across all three resolutions.
    ---------
    Arguments:
    multiresolution_map: The multiresolution map.
    
    Returns:
    multiresolution_map: The weighted multiresolution attention map."""

    # Use weights to normalize the multiresolution map
    suffixes_low = ['-4', '-3', '-2', '-1', '+1', '+2', '+3', '+4']
    for key in multiresolution_map.keys():
        if any(key.endswith(suffix) for suffix in suffixes_low):
            multiresolution_map[key] = [x / 9 for x in multiresolution_map[key]]

    suffixes_med = ['-8', '-7', '-6', '-5', '+5', '+6', '+7', '+8']
    for key in multiresolution_map.keys():
        if any(key.endswith(suffix) for suffix in suffixes_med):
            multiresolution_map[key] = [x / 6 for x in multiresolution_map[key]]
    
    suffixes_high = ['-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '+9', '+10', '+11', '+12', '+13', '+14', '+15', '+16']
    for key in multiresolution_map.keys():
        if any(key.endswith(suffix) for suffix in suffixes_high):
            multiresolution_map[key] = [x / 3 for x in multiresolution_map[key]]
    
    return multiresolution_map

# Attention maps
def attention(variables_depth, max_depth):

    """This function calculates the attention maps for each variable
    in each resolution based on the depth at which they appear in the 
    decision paths.
    ---------
    Arguments:
    variables_depth: The variables and their depth in each path across all trees.
    max_depth: The maximum depth of the decision paths.
    normalize: Whether to normalize the attention maps or not.

    Returns:
    attention_am: The attention map for the am variable.
    attention_co: The attention map for the co variable.
    attention_do: The attention map for the do variable.
    attention_ph: The attention map for the ph variable.
    attention_tu: The attention map for the tu variable.
    attention_wt: The attention map for the wt variable.
    """
    
    # Initialize the resolution names
    resolutions = ['high', 'med', 'low']

    # Initialize the attention maps
    attention = {resolution: {} for resolution in resolutions}

    # Get the number of times each variable appears at each depth
    for resolution in resolutions:
        frequency_depth = np.zeros((len(variables_depth[resolution]), max_depth))
        for i, var in enumerate(variables_depth[resolution].keys()):
            for pos in variables_depth[resolution][var]:
                frequency_depth[i, pos] += 1

        # Convert data to int
        frequency_depth = frequency_depth.astype(int) 

        # Assign the data to the dictionary with the same keys
        for var in variables_depth[resolution].keys():
            attention[resolution][var] = []
        
        for i, var in enumerate(variables_depth[resolution].keys()):
            for j in range(max_depth):
                attention[resolution][var].append(frequency_depth[i, j])

    # Normalize the attention maps
    attention = normalizer(attention)

    # Initialize the multiresolution attention map
    attention_multiresolution = {}
    for key in attention['high'].keys():
        attention_multiresolution[key] = [0] * max_depth

    # Perform element-wise addition of the attention maps to obtain the total attention
    for resolution in resolutions:
        for key in attention[resolution]:
            for i in range(max_depth):
                attention_multiresolution[key][i] += attention[resolution][key][i]

    # Use weights to normalize the multiresolution attention map
    attention_multiresolution = weighter(attention_multiresolution)

    # Subset the attention for each variable
    attention_am = {key: value for key, value in attention_multiresolution.items() if key.startswith('am')}
    attention_co = {key: value for key, value in attention_multiresolution.items() if key.startswith('co')}
    attention_do = {key: value for key, value in attention_multiresolution.items() if key.startswith('do')}
    attention_ph = {key: value for key, value in attention_multiresolution.items() if key.startswith('ph')}
    attention_tu = {key: value for key, value in attention_multiresolution.items() if key.startswith('tu')}
    attention_wt = {key: value for key, value in attention_multiresolution.items() if key.startswith('wt')}

    return attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt

# Multivariate attention map
def multivariate_attention(attention_am, attention_co, attention_do, attention_ph, attention_tu, attention_wt):

    """This function calculates the multivariate attention map for all variables
    based on the attention maps of each variable.
    ---------
    Arguments:
    attention_am: The attention map for the am variable.
    attention_co: The attention map for the co variable.
    attention_do: The attention map for the do variable.
    attention_ph: The attention map for the ph variable.
    attention_tu: The attention map for the tu variable.
    attention_wt: The attention map for the wt variable.

    Returns:
    attention_total: The total attention map for all variables.
    """

    # Replace the keys to be the 2: items
    attention_am = {key[2:]: value for key, value in attention_am.items()}
    attention_co = {key[2:]: value for key, value in attention_co.items()}
    attention_do = {key[2:]: value for key, value in attention_do.items()}
    attention_ph = {key[2:]: value for key, value in attention_ph.items()}
    attention_tu = {key[2:]: value for key, value in attention_tu.items()}
    attention_wt = {key[2:]: value for key, value in attention_wt.items()}
    
    # Perform element-wise addition of the attention maps to obtain the total attention
    attention_total = {key: [sum(x) for x in zip(attention_am[key], attention_co[key], attention_do[key], attention_ph[key], attention_tu[key], attention_wt[key])] for key in attention_am.keys()}

    return attention_total

# Thresholds
def thresholds(variables_thresholds):

    """This function calculates the number of times the thresholds are within
    the inervals, for each variable.
    ---------
    Arguments:
    variables_thresholds: The variables and their threshold in each path across all trees.
    normalize: Whether to normalize the thresholds or not.

    Returns:
    thresholds_am: The thresholds for the am variable.
    thresholds_co: The thresholds for the co variable.
    thresholds_do: The thresholds for the do variable.
    thresholds_ph: The thresholds for the ph variable.
    thresholds_tu: The thresholds for the tu variable.
    thresholds_wt: The thresholds for the wt variable.
    """
    
    # Initialize the resolution names
    resolutions = ['high', 'med', 'low']

    # Define the intervals for the thresholds [0, 0.05), [0.05, 0.1), ..., [0.95, 1]
    intervals_thresholds = [(i / 20, (i + 1) / 20) for i in range(20)]

    # Initialize the dictionary to store the thresholds for each variable
    variables_thresholds_bins = {resolutions: {} for resolutions in resolutions}

    for resolution in resolutions:
        variables_thresholds_bins[resolution] = {key: [0] * len(intervals_thresholds) for key in variables_thresholds[resolution].keys()}
        for i, (start, end) in enumerate(intervals_thresholds):
            for var in variables_thresholds[resolution].keys():
                for threshold in variables_thresholds[resolution][var]:
                    if start <= threshold < end:
                        variables_thresholds_bins[resolution][var][i] += 1
    
    # Normalize the thresholds
    variables_thresholds_bins = normalizer(variables_thresholds_bins)

    # Initialize the multiresolution threshold maps
    thresholds_multiresolution = {}
    for key in variables_thresholds_bins['high'].keys():
        thresholds_multiresolution[key] = [0] * len(intervals_thresholds)

    # Perform element-wise addition of the thresholds to obtain the total thresholds
    for resolution in resolutions:
        for key in variables_thresholds_bins[resolution]:
            for i in range(len(intervals_thresholds)):
                thresholds_multiresolution[key][i] += variables_thresholds_bins[resolution][key][i]
    
    # Use weights to normalize the multiresolution threshold map
    thresholds_multiresolution = weighter(thresholds_multiresolution)

    # Subset the thresholds for each variable
    thresholds_am = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('am')}
    thresholds_co = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('co')}
    thresholds_do = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('do')}
    thresholds_ph = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('ph')}
    thresholds_tu = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('tu')}
    thresholds_wt = {key: value for key, value in thresholds_multiresolution.items() if key.startswith('wt')}

    return thresholds_am, thresholds_co, thresholds_do, thresholds_ph, thresholds_tu, thresholds_wt

# Distances
def distances(variables_distances):

    """This function calculates the number of times the distances between threshold and value
    are within the inervals, for each variable.
    ---------
    Arguments:
    variables_distances: The variables and their distances in each path across all trees.
    normalize: Whether to normalize the distances or not.

    Returns:
    distances_am: The distances for the am variable.
    distances_co: The distances for the co variable.
    distances_do: The distances for the do variable.
    distances_ph: The distances for the ph variable.
    distances_tu: The distances for the tu variable.
    distances_wt: The distances for the wt variable.
    """

    # Initialize the resolution names
    resolutions = ['high', 'med', 'low']
    
    # Define the intervals for the distances [-1, -0.9), [-0.9, -0.8), ..., [0.9, 1]
    intervals_distances = [(i / 10, (i + 1) / 10) for i in range(-10, 10)]

    # Initialize the dictionary to store the distances for each variable
    variables_distances_bins = {resolutions: {} for resolutions in resolutions}

    for resolution in resolutions:
        variables_distances_bins[resolution] = {key: [0] * len(intervals_distances) for key in variables_distances[resolution].keys()}
        for i, (start, end) in enumerate(intervals_distances):
            for var in variables_distances[resolution].keys():
                for distance in variables_distances[resolution][var]:
                    if start <= distance < end:
                        variables_distances_bins[resolution][var][i] += 1
    
    # Normalize the distances
    variables_distances_bins = normalizer(variables_distances_bins)

    # Initialize the multiresolution distance maps
    distances_multiresolution = {}
    for key in variables_distances_bins['high'].keys():
        distances_multiresolution[key] = [0] * len(intervals_distances)
    
    # Perform element-wise addition of the distances to obtain the total distances
    for resolution in resolutions:
        for key in variables_distances_bins[resolution]:
            for i in range(len(intervals_distances)):
                distances_multiresolution[key][i] += variables_distances_bins[resolution][key][i]
    
    # Use weights to normalize the multiresolution distance map
    distances_multiresolution = weighter(distances_multiresolution)
    
    # Subset the distances for each variable
    distances_am = {key: value for key, value in distances_multiresolution.items() if key.startswith('am')}
    distances_co = {key: value for key, value in distances_multiresolution.items() if key.startswith('co')}
    distances_do = {key: value for key, value in distances_multiresolution.items() if key.startswith('do')}
    distances_ph = {key: value for key, value in distances_multiresolution.items() if key.startswith('ph')}
    distances_tu = {key: value for key, value in distances_multiresolution.items() if key.startswith('tu')}
    distances_wt = {key: value for key, value in distances_multiresolution.items() if key.startswith('wt')}

    return distances_am, distances_co, distances_do, distances_ph, distances_tu, distances_wt

# Kullback-Leibler divergence between two attention maps
def kl_divergence(attention_1, attention_2):
    
    """This function calculates the Kulback-Leibler divergence between
    two attention maps.
    ---------
    Arguments:
    attention_1 (dict): The first attention map.
    attention_2 (dict): The second attention map.

    Returns:
    kl_divergence (float): The Kulback-Leibler divergence between the two attention maps.
    """
    
    # Convert the attention maps to numpy arrays
    attention_1 = np.array(list(attention_1.values()))
    attention_2 = np.array(list(attention_2.values()))

    # Check if the attention maps have the same shape
    if attention_1.shape != attention_2.shape:
        
        # Check which needs padding and add it
        if attention_1.shape[1] > attention_2.shape[1]:
            attention_2 = np.pad(attention_2, ((0, 0), (0, attention_1.shape[1] - attention_2.shape[1])), mode='constant')
        else:
            attention_1 = np.pad(attention_1, ((0, 0), (0, attention_2.shape[1] - attention_1.shape[1])), mode='constant')

    # Normalize the attention maps (heatmaps) to ensure they sum to 1
    attention_1 = attention_1 / np.sum(attention_1)
    attention_2 = attention_2 / np.sum(attention_2)

    # Add a small value epsilon to ensure no division by zero
    epsilon = np.finfo(float).eps
    attention_1 = attention_1 + epsilon
    attention_2 = attention_2 + epsilon

    # Compute Kullback-Leibler divergence
    kl_divergence = np.sum(attention_1 * np.log(attention_1 / attention_2))

    return kl_divergence

def attention_plotter(attention_maps, event_number, station, type):

    """This function plots the attention maps for each variable.
    ---------
    Arguments:
    attention_maps: The attention maps for each variable.

    Returns:
    None."""

    variables = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
    
    # Define the plot
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Univariate path attention', fontname='Arial', fontsize=25)

    for attention_map, ax, var in zip(attention_maps, axs.flat, variables):
        sns.heatmap(pd.DataFrame(attention_map).T, cmap='Reds', cbar=False, ax=ax)
        ax.set_xlabel('Tree depth', fontsize=21) # It could also be Contextual observations
        ax.set_ylabel('Time window index', fontsize=21)
        ax.tick_params(axis='both', which='both', rotation=0, labelsize=19)
        ax.set_title(f'{var}', fontname='Arial', fontsize=21)
        
        # Adjust the x-axis ticks to avoid overlap depending on the maximum depth
        if len(list(attention_map.values())[0]) >= 10:
            tick_interval = 2
            ax.set_xticks(np.arange(0.5, len(list(attention_map.values())[0]), tick_interval))
        if len(list(attention_map.values())[0]) >= 18:
            tick_interval = 4
            ax.set_xticks(np.arange(0.5, len(list(attention_map.values())[0]), tick_interval))
    
    plt.tight_layout()
    # plt.show()

    # Save figure
    plt.savefig(f'results/attention_maps_{station}_{type}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

def multivariate_attention_plotter(attention_total, event_number, station, type):

    """This function plots the multivariate attention map for all variables.
    ---------
    Arguments:
    attention_total: The total attention map for all variables.

    Returns:
    None."""

    # Define the plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(pd.DataFrame(attention_total).T, cmap='Reds', cbar=False)
    plt.xlabel('Tree depth', fontsize=21)
    plt.ylabel('Time window index', fontsize=21)
    plt.tick_params(axis='both', which='both', labelsize=19)
    plt.title('Multivariate path attention', fontname='Arial', fontsize=25)
    # plt.show()

    # Adjust the x-axis ticks to avoid overlap depending on the maximum depth
    if len(list(attention_total.values())[0]) >= 10:
        tick_interval = 2
        plt.xticks(np.arange(0.5, len(list(attention_total.values())[0]), tick_interval))
    if len(list(attention_total.values())[0]) >= 18:
        tick_interval = 4
        plt.xticks(np.arange(0.5, len(list(attention_total.values())[0]), tick_interval))
    
    # Adjust the y-axis ticks to avoid overlap depending on the number of variables
    tick_interval = 2
    plt.yticks(np.arange(0.5, len(list(attention_total.keys())), tick_interval), list(attention_total.keys())[::tick_interval])
    
    # Save figure
    plt.savefig(f'results/multivariate_attention_map_{station}_{type}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

def threshold_plotter(threshold_maps, event_number, station, type):
    
    """This function plots the thresholds for each variable.
    ---------
    Arguments:
    threshold_maps: The thresholds for each variable.

    Returns:
    None."""

    variables = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']

    # Define the plot
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Univariate learned thresholds', fontname='Arial', fontsize=25)

    for threshold_map, ax, var in zip(threshold_maps, axs.flat, variables):
        sns.heatmap(pd.DataFrame(threshold_map).T, cmap='Greens', cbar=False, ax=ax)
        ax.set_xlabel('Threshold value', fontsize=21)
        ax.set_ylabel('Time window index', fontsize=21)
        ax.tick_params(axis='both', which='both', labelsize=19)
        ax.set_title(f'{var}', fontname='Arial', fontsize=21)

        # Generate ticks from 0 to 1 with steps of 0.05
        ticks = np.arange(0, 1.05, 0.05)
        
        # Calculate the equivalent positions for the ticks
        tick_positions = np.linspace(0, pd.DataFrame(threshold_map).T.shape[1], len(ticks))
        
        # Set the ticks and labels on the x-axis, showing every fourth tick
        ax.set_xticks(tick_positions[::4])
        ax.set_xticklabels(np.round(ticks[::4], 2))

    plt.tight_layout()
    # plt.show()

    # Save figure
    plt.savefig(f'results/threshold_maps_{station}_{type}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

def distance_plotter(distance_maps, event_number, station, type):

    """This function plots the distances for each variable.
    ---------
    Arguments:
    distance_maps: The distances for each variable.

    Returns:
    None."""

    variables = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']

    # Define the plot
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Univariate distance to threshold', fontname='Arial', fontsize=25)

    for threshold_map, ax, var in zip(distance_maps, axs.flat, variables):
        sns.heatmap(pd.DataFrame(threshold_map).T, cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Distance to threshold', fontsize=21)
        ax.set_ylabel('Time window index', fontsize=21)
        ax.tick_params(axis='both', which='both', labelsize=19)
        ax.set_title(f'{var}', fontname='Arial', fontsize=21)

        # Generate ticks from 0 to 1 with steps of 0.05
        ticks = np.arange(-1, 1.1, 0.1)
        
        # Calculate the equivalent positions for the ticks
        tick_positions = np.linspace(0, pd.DataFrame(threshold_map).T.shape[1], len(ticks))
        
        # Set the ticks and labels on the x-axis, showing every fourth tick
        ax.set_xticks(tick_positions[::4])
        ax.set_xticklabels(np.round(ticks[::4], 2))

    plt.tight_layout()
    # plt.show()

    # Save figure
    plt.savefig(f'results/distance_maps_{station}_{type}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')

def kl_plotter(kl_distances, event_number, station, data_type):

    """This function plots the Kullback-Leibler divergence distributions.
    ---------
    Arguments:
    kl_distances: The Kullback-Leibler divergence distributions.
    event_number: The event number.
    station: The station name.
    data_type: The data type.

    Returns:
    None."""

    # Plot the KL divergences
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(kl_distances[0], color='lightcoral', label='Labeled anomalies', linewidth=1, fill=True, ax=ax)
    sns.kdeplot(kl_distances[1], color='limegreen', label='Detected anomalies', linewidth=1, fill=True, ax=ax)
    sns.kdeplot(kl_distances[2], color='cornflowerblue', label='Background', linewidth=1, fill=True, ax=ax)

    ax.set_title('Kullback-Leibler divergence distributions', fontfamily='serif', fontsize=23)
    ax.set_xlabel('Divergence', fontsize=21)
    ax.set_ylabel('Density', fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=19)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=3, fontsize=19)
    # plt.show()

    plt.savefig(f'results/kl_divergence_{station}_{data_type[:2]}_{event_number}.pdf', format='pdf', dpi=300, bbox_inches='tight')    

# Decision paths plot
def dp_plotter(data, model, resolution, station, name):

    """NOT CURRENTLY USED.
    This function explains the decision of a Random Forest model
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
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt+15',
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
    sns.heatmap(heatmap_data, xticklabels=range(max_len), yticklabels=list(variables.keys()), 
                cmap='viridis', annot=True, annot_kws={"size": 14}, fmt="d") # annot size 14 for anomalies and 10 for background
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Position', fontsize=16)
    plt.ylabel('Variable', fontsize=16)
    # plt.show()

    plt.savefig(f'results/{name}_var.png')

    # Close figure
    plt.close()

    # Variable-threshold plot
    # Read the data and get the mean for each variable
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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Position', fontsize=16)
    plt.ylabel('Variable', fontsize=16)
    # plt.show()

    plt.savefig(f'results/{name}_thre.png')

    # Close figure
    plt.close()

# Mean-position plot
def mean_plotter(data, resolution, num_variables, station, name):

    """NOT CURRENTLY USED.
    This function plots the distance between the mean of each variable
    and the values of the variables in the data window.
    ---------
    Arguments:
    data: The data to be explained.
    resolution: The resolution of the model.
    num_variables: The number of variables in the data.
    station: The station number.
    name: The title of the plot.

    Returns:
    None.
    """
    
    # Read the data and get the mean for each variable
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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Position', fontsize=16)
    plt.ylabel('Variable', fontsize=16)
    # plt.show()

    plt.savefig(f'results/{name}_mean.png')

    # Close figure
    plt.close()

def tree_plotter(model, resolution, tree_number):

    """This function plots a tree of a Random Forest model.
    ---------
    Arguments:
    model: The Random Forest model to plot.
    resolution: The resolution of the model.
    tree_number: The number of the tree to plot.

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
                    'am+15', 'co+15', 'do+15', 'ph+15', 'tu+15', 'wt+15',
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

    # Plot the tree using the export_graphviz function
    export_graphviz(model.estimators_[tree_number], out_file='tree.dot',
                    feature_names=feature_names, # Number of data points in a window
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)

    # Convert to png using system command (requires Graphviz, which has to be installed, added to PATH, and pip installed)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', f'tree_{resolution}_{tree_number}.png', '-Gdpi=600'])

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
