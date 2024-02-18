import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from scipy.interpolate import interp1d

station = 906

# Read the data
df_original = pd.read_csv(f'plots/{station}_plots.csv', delimiter=',', index_col=['iteration'])

# Define figure
fig = plt.figure(figsize=(6, 6))
ax = plt.axes()

# Filter the data for 'high' resolution
df = df_original[df_original['resolution'] == 'high']

xnew = np.linspace(0, df.index[-1], num=500, endpoint=True) # The second parameter affects the length of the data when plotted

# Plot number of background windows
accuracy = interp1d(list(df.index), list(df['background_windows']), kind='cubic')
ax.plot(xnew, accuracy(xnew), color='#030aa7', label='Background windows high')

# Plot number of anomaly windows
error_rate = interp1d(list(df.index), list(df['anomalous_windows']), kind='cubic')
ax.plot(xnew, error_rate(xnew), color='#8f1402', label='Anomalous windows high')

# Filter the data for 'med' resolution
df = df_original[df_original['resolution'] == 'med']

xnew = np.linspace(0, df.index[-1], num=500, endpoint=True) # The second parameter affects the length of the data when plotted

# Plot number of background windows
accuracy = interp1d(list(df.index), list(df['background_windows']), kind='cubic')
ax.plot(xnew, accuracy(xnew), color='#247afd', label='Background windows med')

# Plot number of anomaly windows
error_rate = interp1d(list(df.index), list(df['anomalous_windows']), kind='cubic')
ax.plot(xnew, error_rate(xnew), color='red', label='Anomalous windows med')

# Filter the data for 'low' resolution
df = df_original[df_original['resolution'] == 'low']

xnew = np.linspace(0, df.index[-1], num=500, endpoint=True) # The second parameter affects the length of the data when plotted

# Plot number of background windows
accuracy = interp1d(list(df.index), list(df['background_windows']), kind='cubic')
ax.plot(xnew, accuracy(xnew), color='#a2bffe', label='Background windows high')

# Plot number of anomaly windows
error_rate = interp1d(list(df.index), list(df['anomalous_windows']), kind='cubic')
ax.plot(xnew, error_rate(xnew), color='#fc5a50', label='Anomalous windows med')

# Define axes limits, title and labels
ax.set(xlim=(df.index[0]-1, df.index[-1]+1), ylim=(-0.2, max(list(df['background_windows']))*1.2),
    # title='Number windows 901',
    xlabel='Iteration', ylabel='Number of windows')

# Change the fontsize of the axis titles
ax.set_xlabel('Iteration', fontsize=16)
ax.set_ylabel('Score', fontsize=16)
ax.set_title(f'Window number change {station}', fontsize=18, loc='right')

# Change the fontsize of the ticks
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add legend
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


plt.savefig(f'plots/{station}_plots.png', dpi=300)

