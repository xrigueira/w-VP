import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from scipy.interpolate import interp1d

file_name = 'turbidity_ph.csv'
df = pd.read_csv(file_name, delimiter=';')

# Define figure
fig = plt.figure(figsize=(9,5))
ax = plt.axes()

# Add the data
smoothed = 'Yes'

if smoothed == 'No':
    ax.plot(df.index, df['Scaled_turbidity'], color='blue', marker='o', label='Turbidity')
    ax.plot(df.index, df['Scaled_pH'], color='red', marker='o', label='pH')

elif smoothed == 'Yes':
    xnew = np.linspace(0, 21, num=500, endpoint=True) # The second parameter affects the length of the data when plotted

    var1 = interp1d(list(df.index), list(df['Scaled_turbidity']), kind='cubic')
    ax.plot(xnew, var1(xnew), color='darkviolet', label='Turbidity')
    var2 = interp1d(list(df.index), list(df['Scaled_pH']), kind='cubic')
    ax.plot(xnew, var2(xnew), color='red', label='pH')
    
    # # In the case of the database with three variables
    # var3 = interp1d(list(df.index), list(df['Scaled_turbidity']), kind='cubic')
    # ax.plot(xnew, var3(xnew), color='darkviolet', label='Turbidity')

# Define axes limits, title and labels
ax.set(xlim=(df.index[0]-1, df.index[-1]+1), ylim=(-0.2, 1.2),
    title='Turbidity v. pH',
    xlabel='Outlier number', ylabel='Scaled mean value')


# Add legend
plt.legend()

plt.show()
