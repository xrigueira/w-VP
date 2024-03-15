import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Read the data and select those columns with the desired variables
data_901 = pd.read_csv(f'data/labeled_901_smo.csv').iloc[:, 1:-2]
data_905 = pd.read_csv(f'data/labeled_905_smo.csv').iloc[:, 1:-2]
data_907 = pd.read_csv(f'data/labeled_907_smo.csv').iloc[:, 1:-2]

# Concatenating horizontally
data = pd.concat([data_901, data_905, data_907], axis=1)

# Sorting columns alphabetically
data = data.reindex(sorted(data.columns), axis=1)

# Assigning new names to columns
new_column_names = ['am-901', 'am-905', 'am-907', 
                    'co-901', 'co-905', 'co-907', 
                    'do-901', 'do-905', 'do-907', 
                    'ph-901', 'ph-905', 'ph-907',
                    'tu-901', 'tu-905', 'tu-907',
                    'wt-901', 'wt-905', 'wt-907']

data.columns = new_column_names

# Plotting the boxplots
plt.figure(figsize=(12, 8))
sns.violinplot(data=data)
plt.xticks(rotation=45)

plt.savefig('plots/violinplot.png')

plt.show()