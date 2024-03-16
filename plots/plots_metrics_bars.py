import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('ggplot')

# Read the data
data = pd.read_csv(f'plots/data_plots.csv', delimiter=',', index_col=['station'])

# Extract the required columns
stations = tuple(data.index)
metrics = {
    'Accuracy': tuple(data['accuracy']),
    'Error rate': tuple(data['error_rate']),
    'Precision': tuple(data['precision']),
    'Recall': tuple(data['recall'])
}

colors = ['salmon', 'cornflowerblue', 'darkseagreen', 'khaki']  # Define your own colors here

x = np.arange(len(stations))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, metric, color in zip(metrics.keys(), metrics.values(), colors):
    offset = width * multiplier
    rects = ax.bar(x + offset, metric, width, label=attribute, color=color)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score', fontsize=16)
ax.set_xticks(x + width * 1.5, stations)
ax.legend(loc='upper center', bbox_to_anchor=(0.58, 1.1), ncol=4)
ax.set_ylim(0, 1)

# Change the fontsize of the ticks
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.savefig('plots/metrics_bars.png', bbox_inches='tight')

# plt.show()