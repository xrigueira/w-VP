import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

variables = [['co-7', 'am-6'], ['co+3', 'am+7'], ['co+4', 'am-8'], ['am-4'], ['am-8'], ['am-4'], ['am+0'], ['am+3'], ['am-4'], ['am+4'], ['am+3'], ['ph+2', 'am+7'], ['am-3'], ['co-5', 'am+2'], ['am-6'], ['am+1'], ['am-4'], ['co+0', 'am-6']]
thresholds = [[0.39755457639694214, 0.13899371027946472], [0.39161571860313416, 0.13899371027946472], [0.39755457639694214, 0.13899371027946472], [0.13899371027946472], [0.13962264358997345], [0.13899371027946472], [0.13899371027946472], [0.13899371027946472], [0.13899371027946472], [0.13899371027946472], [0.13899371027946472], [0.5359848737716675, 0.13899371027946472], [0.13899371027946472], [0.40096069872379303, 0.13899371027946472], [0.13899371027946472], [0.13962264358997345], [0.13899371027946472], [0.3981659263372421, 0.13899371027946472]]

# Extract variables and their thresholds
variables_dict = {}
for sublist, sublist_thresholds in zip(variables, thresholds):
    for var, threshold in zip(sublist, sublist_thresholds):
        var_type = var.split('-')[0].split('+')[0]  # Extract variable type
        if var_type not in variables_dict:
            variables_dict[var_type] = {}
        if var not in variables_dict[var_type]:
            variables_dict[var_type][var] = []
        variables_dict[var_type][var].append(threshold)

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

    # Create the violin plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='Variable', y='Threshold', data=df, order=df['Variable'].unique())
    plt.title(f'Violin plot for {var_type} variables')
    plt.show()

# Create the violin plot
plt.figure(figsize=(10, 8))
sns.violinplot(x='Variable', y='Threshold', data=df)
plt.show()