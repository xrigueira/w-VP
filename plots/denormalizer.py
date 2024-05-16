# Make a denormalizer plot. It could be a violin plot per variable with a gradient color and the 0 to 1 in the right axis and the left annotated with the actual values. It would be one plot per variable.
# https://stackoverflow.com/questions/57323884/matplotlib-seaborn-violin-plot-with-colormap

import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.preprocessing import MinMaxScaler

# Read the data
station = 901
data_901 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 905
data_905 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 907
data_907 = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

# Get the min and max for each variable
min_ = [data_901.ammonium_901.min(), data_905.ammonium_905.min(), data_907.ammonium_907.min()]
max_ = [data_901.ammonium_901.max(), data_905.ammonium_905.max(), data_907.ammonium_907.max()]

# Check if the min is below 0
for i in range(3):
    if min_[i] < 0:
        min_[i] = 0

# Normalize the data
scaler = MinMaxScaler()
ammonium_901 = scaler.fit_transform(data_901['ammonium_901'].values.reshape(-1, 1))
ammonium_905 = scaler.fit_transform(data_905['ammonium_905'].values.reshape(-1, 1))
ammonium_907 = scaler.fit_transform(data_907['ammonium_907'].values.reshape(-1, 1))

# Convert numpy arrays to pandas Series to concatenate them
ammonium_901 = pd.Series(ammonium_901.flatten())
ammonium_905 = pd.Series(ammonium_905.flatten())
ammonium_907 = pd.Series(ammonium_907.flatten())

# Concatenate data horizontally
data = pd.concat([ammonium_901, ammonium_905, ammonium_907], axis=1)
data.columns = ['am-901', 'am-905', 'am-907']

# Create the figure
plt.figure(figsize=(12, 8))
violin_parts = sns.violinplot(data=data)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)

# Iterate over the violins
for i, violin in enumerate(violin_parts.collections[::1]):
    # Get the min and max value for this violin
    min_val = min_[i]
    max_val = max_[i]

    # Get the position of the violin
    violin_center = violin.get_paths()[0].vertices.mean(0)

    # Add a small offset to the x-coordinate
    offset = 0.05

    # Add the min and max value as labels to the end of the violin
    plt.text(violin_center[0] + offset, 0, f'Min: {min_val}', fontsize=12)
    plt.text(violin_center[0] + offset, 1, f'Max: {max_val}', fontsize=12)


plt.show()