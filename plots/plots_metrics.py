import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from scipy.interpolate import interp1d

station = 901

# Read the data
df_original = pd.read_csv(f'plots/{station}_plots.csv', delimiter=',', index_col=['iteration'])

# Filter the data for 'high' resolution
df = df_original[df_original['resolution'] == 'med']

# Define figure
fig = plt.figure(figsize=(6, 6))
ax = plt.axes()

xnew = np.linspace(0, df.index[-1], num=500, endpoint=True) # The second parameter affects the length of the data when plotted

# Plot accuracy
accuracy = interp1d(list(df.index), list(df['accuracy']), kind='cubic')
ax.plot(xnew, accuracy(xnew), color='gold', label='Accuracy')

# Plot error_rate
error_rate = interp1d(list(df.index), list(df['error_rate']), kind='cubic')
ax.plot(xnew, error_rate(xnew), color='red', label='Error Rate')

# Plot precision
precision = interp1d(list(df.index), list(df['precision']), kind='cubic')
ax.plot(xnew, precision(xnew), color='green', label='Precision')

# Plot recall
recall = interp1d(list(df.index), list(df['recall']), kind='cubic')
ax.plot(xnew, recall(xnew), color='blue', label='Recall')

# Define axes limits, title and labels
ax.set(xlim=(df.index[0]-1, df.index[-1]+1), ylim=(-0.2, 1.2),
    # title='Metrics 901',
    xlabel='Iteration', ylabel='Score')

# Add legend
plt.legend()

plt.show()

# plt.savefig(f'plots/{station}_plots.png', dpi=300)

