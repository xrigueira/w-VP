"""Comapare the distributions of the original data with gaps before and
after being filled."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'

from scipy.stats import ks_2samp

# Read the data
station = 901
data_901_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_901_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 905
data_905_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_905_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

station = 907
data_907_gaps = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
data_907_filled = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

#%% Get the percentage of missing values for each variable and station
# missing_am_901 = data_901_gaps.ammonium_901.isna().sum() / data_901_gaps.shape[0]
# missing_co_901 = data_901_gaps.conductivity_901.isna().sum() / data_901_gaps.shape[0]
# missing_do_901 = data_901_gaps.dissolved_oxygen_901.isna().sum() / data_901_gaps.shape[0]
# missing_ph_901 = data_901_gaps.pH_901.isna().sum() / data_901_gaps.shape[0]
# missing_tu_901 = data_901_gaps.turbidity_901.isna().sum() / data_901_gaps.shape[0]
# missing_wt_901 = data_901_gaps.water_temperature_901.isna().sum() / data_901_gaps.shape[0]

# missing_am_905 = data_905_gaps.ammonium_905.isna().sum() / data_905_gaps.shape[0]
# missing_co_905 = data_905_gaps.conductivity_905.isna().sum() / data_905_gaps.shape[0]
# missing_do_905 = data_905_gaps.dissolved_oxygen_905.isna().sum() / data_905_gaps.shape[0]
# missing_ph_905 = data_905_gaps.pH_905.isna().sum() / data_905_gaps.shape[0]
# missing_tu_905 = data_905_gaps.turbidity_905.isna().sum() / data_905_gaps.shape[0]
# missing_wt_905 = data_905_gaps.water_temperature_905.isna().sum() / data_905_gaps.shape[0]

# missing_am_907 = data_907_gaps.ammonium_907.isna().sum() / data_907_gaps.shape[0]
# missing_co_907 = data_907_gaps.conductivity_907.isna().sum() / data_907_gaps.shape[0]
# missing_do_907 = data_907_gaps.dissolved_oxygen_907.isna().sum() / data_907_gaps.shape[0]
# missing_ph_907 = data_907_gaps.pH_907.isna().sum() / data_907_gaps.shape[0]
# missing_tu_907 = data_907_gaps.turbidity_907.isna().sum() / data_907_gaps.shape[0]
# missing_wt_907 = data_907_gaps.water_temperature_907.isna().sum() / data_907_gaps.shape[0]

# # Store the results in a DataFrame
# missing_values = pd.DataFrame({
#     'station': [901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 907, 907, 907, 907, 907, 907],
#     'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
#                 'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
#     'missing_values': [missing_am_901, missing_co_901, missing_do_901, missing_ph_901, missing_tu_901, missing_wt_901,
#                     missing_am_905, missing_co_905, missing_do_905, missing_ph_905, missing_tu_905, missing_wt_905,
#                     missing_am_907, missing_co_907, missing_do_907, missing_ph_907, missing_tu_907, missing_wt_907]
#     })

# print(missing_values)

#%% Extract the data for each variable and station before and after filling. Remove the NaNs for the original data
am_901_gaps = data_901_gaps.ammonium_901.to_numpy()
am_901_gaps = am_901_gaps[~np.isnan(am_901_gaps)]
co_901_gaps = data_901_gaps.conductivity_901.to_numpy()
co_901_gaps = co_901_gaps[~np.isnan(co_901_gaps)]
do_901_gaps = data_901_gaps.dissolved_oxygen_901.to_numpy()
do_901_gaps = do_901_gaps[~np.isnan(do_901_gaps)]
ph_901_gaps = data_901_gaps.pH_901.to_numpy()
ph_901_gaps = ph_901_gaps[~np.isnan(ph_901_gaps)]
tu_901_gaps = data_901_gaps.turbidity_901.to_numpy()
tu_901_gaps = tu_901_gaps[~np.isnan(tu_901_gaps)]
wt_901_gaps = data_901_gaps.water_temperature_901.to_numpy()
wt_901_gaps = wt_901_gaps[~np.isnan(wt_901_gaps)]

am_901_filled = data_901_filled.ammonium_901.to_numpy()
am_901_filled = am_901_filled[~np.isnan(am_901_filled)]
co_901_filled = data_901_filled.conductivity_901.to_numpy()
co_901_filled = co_901_filled[~np.isnan(co_901_filled)]
do_901_filled = data_901_filled.dissolved_oxygen_901.to_numpy()
do_901_filled = do_901_filled[~np.isnan(do_901_filled)]
ph_901_filled = data_901_filled.pH_901.to_numpy()
ph_901_filled = ph_901_filled[~np.isnan(ph_901_filled)]
tu_901_filled = data_901_filled.turbidity_901.to_numpy()
tu_901_filled = tu_901_filled[~np.isnan(tu_901_filled)]
wt_901_filled = data_901_filled.water_temperature_901.to_numpy()
wt_901_filled = wt_901_filled[~np.isnan(wt_901_filled)]

am_905_gaps = data_905_gaps.ammonium_905.to_numpy()
am_905_gaps = am_905_gaps[~np.isnan(am_905_gaps)]
co_905_gaps = data_905_gaps.conductivity_905.to_numpy()
co_905_gaps = co_905_gaps[~np.isnan(co_905_gaps)]
do_905_gaps = data_905_gaps.dissolved_oxygen_905.to_numpy()
do_905_gaps = do_905_gaps[~np.isnan(do_905_gaps)]
ph_905_gaps = data_905_gaps.pH_905.to_numpy()
ph_905_gaps = ph_905_gaps[~np.isnan(ph_905_gaps)]
tu_905_gaps = data_905_gaps.turbidity_905.to_numpy()
tu_905_gaps = tu_905_gaps[~np.isnan(tu_905_gaps)]
wt_905_gaps = data_905_gaps.water_temperature_905.to_numpy()
wt_905_gaps = wt_905_gaps[~np.isnan(wt_905_gaps)]

am_905_filled = data_905_filled.ammonium_905.to_numpy()
am_905_filled = am_905_filled[~np.isnan(am_905_filled)]
co_905_filled = data_905_filled.conductivity_905.to_numpy()
co_905_filled = co_905_filled[~np.isnan(co_905_filled)]
do_905_filled = data_905_filled.dissolved_oxygen_905.to_numpy()
do_905_filled = do_905_filled[~np.isnan(do_905_filled)]
ph_905_filled = data_905_filled.pH_905.to_numpy()
ph_905_filled = ph_905_filled[~np.isnan(ph_905_filled)]
tu_905_filled = data_905_filled.turbidity_905.to_numpy()
tu_905_filled = tu_905_filled[~np.isnan(tu_905_filled)]
wt_905_filled = data_905_filled.water_temperature_905.to_numpy()
wt_905_filled = wt_905_filled[~np.isnan(wt_905_filled)]

am_907_gaps = data_907_gaps.ammonium_907.to_numpy()
am_907_gaps = am_907_gaps[~np.isnan(am_907_gaps)]
co_907_gaps = data_907_gaps.conductivity_907.to_numpy()
co_907_gaps = co_907_gaps[~np.isnan(co_907_gaps)]
do_907_gaps = data_907_gaps.dissolved_oxygen_907.to_numpy()
do_907_gaps = do_907_gaps[~np.isnan(do_907_gaps)]
ph_907_gaps = data_907_gaps.pH_907.to_numpy()
ph_907_gaps = ph_907_gaps[~np.isnan(ph_907_gaps)]
tu_907_gaps = data_907_gaps.turbidity_907.to_numpy()
tu_907_gaps = tu_907_gaps[~np.isnan(tu_907_gaps)]
wt_907_gaps = data_907_gaps.water_temperature_907.to_numpy()
wt_907_gaps = wt_907_gaps[~np.isnan(wt_907_gaps)]

am_907_filled = data_907_filled.ammonium_907.to_numpy()
am_907_filled = am_907_filled[~np.isnan(am_907_filled)]
co_907_filled = data_907_filled.conductivity_907.to_numpy()
co_907_filled = co_907_filled[~np.isnan(co_907_filled)]
do_907_filled = data_907_filled.dissolved_oxygen_907.to_numpy()
do_907_filled = do_907_filled[~np.isnan(do_907_filled)]
ph_907_filled = data_907_filled.pH_907.to_numpy()
ph_907_filled = ph_907_filled[~np.isnan(ph_907_filled)]
tu_907_filled = data_907_filled.turbidity_907.to_numpy()
tu_907_filled = tu_907_filled[~np.isnan(tu_907_filled)]
wt_907_filled = data_907_filled.water_temperature_907.to_numpy()
wt_907_filled = wt_907_filled[~np.isnan(wt_907_filled)]

#%% Plot the empirical cumulative distributions of each variable comparing the data with gaps and the filled version
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 8))

sns.ecdfplot(data=am_901_gaps, color='lightcoral', ax=axes[0, 0])
sns.ecdfplot(data=am_901_filled, label='red', ax=axes[0, 0])

sns.ecdfplot(data=co_901_gaps, color='cornflowerblue', ax=axes[0, 1])
sns.ecdfplot(data=co_901_filled, color='blue', ax=axes[0, 1])

sns.ecdfplot(data=do_901_gaps, color='mediumpurple', ax=axes[0, 2])
sns.ecdfplot(data=do_901_filled, color='purple', ax=axes[0, 2])

sns.ecdfplot(data=ph_901_gaps, color='dimgray', ax=axes[0, 3])
sns.ecdfplot(data=ph_901_filled, color='darkgray', ax=axes[0, 3])

sns.ecdfplot(data=tu_901_gaps, color='gold', ax=axes[0, 4])
sns.ecdfplot(data=tu_901_filled, color='goldenrod', ax=axes[0, 4])

sns.ecdfplot(data=wt_901_gaps, color='limegreen', ax=axes[0, 5])
sns.ecdfplot(data=wt_901_filled, color='green', ax=axes[0, 5])

sns.ecdfplot(data=am_905_gaps, color='lightcoral', ax=axes[1, 0])
sns.ecdfplot(data=am_905_filled, color='red', ax=axes[1, 0])

sns.ecdfplot(data=co_905_gaps, color='cornflowerblue', ax=axes[1, 1])
sns.ecdfplot(data=co_905_filled, color='blue', ax=axes[1, 1])

sns.ecdfplot(data=do_905_gaps, color='mediumpurple', ax=axes[1, 2])
sns.ecdfplot(data=do_905_filled, color='purple', ax=axes[1, 2])

sns.ecdfplot(data=ph_905_gaps, color='dimgray', ax=axes[1, 3])
sns.ecdfplot(data=ph_905_filled, color='darkgray', ax=axes[1, 3])

sns.ecdfplot(data=tu_905_gaps, color='gold', ax=axes[1, 4])
sns.ecdfplot(data=tu_905_filled, color='goldenrod', ax=axes[1, 4])

sns.ecdfplot(data=wt_905_gaps, color='limegreen', ax=axes[1, 5])
sns.ecdfplot(data=wt_905_filled, color='green', ax=axes[1, 5])

sns.ecdfplot(data=am_907_gaps, color='lightcoral', ax=axes[2, 0])
sns.ecdfplot(data=am_907_filled, color='red', ax=axes[2, 0])

sns.ecdfplot(data=co_907_gaps, color='cornflowerblue', ax=axes[2, 1])
sns.ecdfplot(data=co_907_filled, color='blue', ax=axes[2, 1])

sns.ecdfplot(data=do_907_gaps, color='mediumpurple', ax=axes[2, 2])
sns.ecdfplot(data=do_907_filled, color='purple', ax=axes[2, 2])

sns.ecdfplot(data=ph_907_gaps, color='dimgray', ax=axes[2, 3])
sns.ecdfplot(data=ph_907_filled, color='darkgray', ax=axes[2, 3])

sns.ecdfplot(data=tu_907_gaps, color='gold', ax=axes[2, 4])
sns.ecdfplot(data=tu_907_filled, color='goldenrod', ax=axes[2, 4])

sns.ecdfplot(data=wt_907_gaps, color='limegreen', ax=axes[2, 5])
sns.ecdfplot(data=wt_907_filled, color='green', ax=axes[2, 5])

# Clean defaul y label and reduce font size for all axes
for ax in axes.flat:
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=6)

# Set the title for each variable
var_names = ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature']
for i, ax in enumerate(axes[0]):
    ax.set_title(var_names[i], fontfamily='serif', fontsize=16)

# Set the y label for each variable
stations = [901, 905, 907]
for i, ax in enumerate(axes):
    ax[0].set_ylabel(stations[i], fontfamily='serif', fontsize=16)

fig.suptitle('Empirical distributions before and after imputation', fontfamily='serif', fontsize=18)
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig('plots/imputation.pdf', format='pdf', dpi=300, bbox_inches='tight')

#%% Get the mean and standard deviation for each variable and station before and after filling
mean_901_am_gaps, std_901_am_gaps, mean_901_am_filled, std_901_am_filled = np.mean(am_901_gaps), np.std(am_901_gaps), np.mean(am_901_filled), np.std(am_901_filled)
mean_901_co_gaps, std_901_co_gaps, mean_901_co_filled, std_901_co_filled = np.mean(co_901_gaps), np.std(co_901_gaps), np.mean(co_901_filled), np.std(co_901_filled)
mean_901_do_gaps, std_901_do_gaps, mean_901_do_filled, std_901_do_filled = np.mean(do_901_gaps), np.std(do_901_gaps), np.mean(do_901_filled), np.std(do_901_filled)
mean_901_ph_gaps, std_901_ph_gaps, mean_901_ph_filled, std_901_ph_filled = np.mean(ph_901_gaps), np.std(ph_901_gaps), np.mean(ph_901_filled), np.std(ph_901_filled)
mean_901_tu_gaps, std_901_tu_gaps, mean_901_tu_filled, std_901_tu_filled = np.mean(tu_901_gaps), np.std(tu_901_gaps), np.mean(tu_901_filled), np.std(tu_901_filled)
mean_901_wt_gaps, std_901_wt_gaps, mean_901_wt_filled, std_901_wt_filled = np.mean(wt_901_gaps), np.std(wt_901_gaps), np.mean(wt_901_filled), np.std(wt_901_filled)

mean_905_am_gaps, std_905_am_gaps, mean_905_am_filled, std_905_am_filled = np.mean(am_905_gaps), np.std(am_905_gaps), np.mean(am_905_filled), np.std(am_905_filled)
mean_905_co_gaps, std_905_co_gaps, mean_905_co_filled, std_905_co_filled = np.mean(co_905_gaps), np.std(co_905_gaps), np.mean(co_905_filled), np.std(co_905_filled)
mean_905_do_gaps, std_905_do_gaps, mean_905_do_filled, std_905_do_filled = np.mean(do_905_gaps), np.std(do_905_gaps), np.mean(do_905_filled), np.std(do_905_filled)
mean_905_ph_gaps, std_905_ph_gaps, mean_905_ph_filled, std_905_ph_filled = np.mean(ph_905_gaps), np.std(ph_905_gaps), np.mean(ph_905_filled), np.std(ph_905_filled)
mean_905_tu_gaps, std_905_tu_gaps, mean_905_tu_filled, std_905_tu_filled = np.mean(tu_905_gaps), np.std(tu_905_gaps), np.mean(tu_905_filled), np.std(tu_905_filled)
mean_905_wt_gaps, std_905_wt_gaps, mean_905_wt_filled, std_905_wt_filled = np.mean(wt_905_gaps), np.std(wt_905_gaps), np.mean(wt_905_filled), np.std(wt_905_filled)

mean_907_am_gaps, std_907_am_gaps, mean_907_am_filled, std_907_am_filled = np.mean(am_907_gaps), np.std(am_907_gaps), np.mean(am_907_filled), np.std(am_907_filled)
mean_907_co_gaps, std_907_co_gaps, mean_907_co_filled, std_907_co_filled = np.mean(co_907_gaps), np.std(co_907_gaps), np.mean(co_907_filled), np.std(co_907_filled)
mean_907_do_gaps, std_907_do_gaps, mean_907_do_filled, std_907_do_filled = np.mean(do_907_gaps), np.std(do_907_gaps), np.mean(do_907_filled), np.std(do_907_filled)
mean_907_ph_gaps, std_907_ph_gaps, mean_907_ph_filled, std_907_ph_filled = np.mean(ph_907_gaps), np.std(ph_907_gaps), np.mean(ph_907_filled), np.std(ph_907_filled)
mean_907_tu_gaps, std_907_tu_gaps, mean_907_tu_filled, std_907_tu_filled = np.mean(tu_907_gaps), np.std(tu_907_gaps), np.mean(tu_907_filled), np.std(tu_907_filled)
mean_907_wt_gaps, std_907_wt_gaps, mean_907_wt_filled, std_907_wt_filled = np.mean(wt_907_gaps), np.std(wt_907_gaps), np.mean(wt_907_filled), np.std(wt_907_filled)

# Store the results in a DataFrame
results = pd.DataFrame({
    'station': [901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'mean_gaps': [mean_901_am_gaps, mean_901_co_gaps, mean_901_do_gaps, mean_901_ph_gaps, mean_901_tu_gaps, mean_901_wt_gaps,
                mean_905_am_gaps, mean_905_co_gaps, mean_905_do_gaps, mean_905_ph_gaps, mean_905_tu_gaps, mean_905_wt_gaps,
                mean_907_am_gaps, mean_907_co_gaps, mean_907_do_gaps, mean_907_ph_gaps, mean_907_tu_gaps, mean_907_wt_gaps],
    'std_gaps': [std_901_am_gaps, std_901_co_gaps, std_901_do_gaps, std_901_ph_gaps, std_901_tu_gaps, std_901_wt_gaps,
                std_905_am_gaps, std_905_co_gaps, std_905_do_gaps, std_905_ph_gaps, std_905_tu_gaps, std_905_wt_gaps,
                std_907_am_gaps, std_907_co_gaps, std_907_do_gaps, std_907_ph_gaps, std_907_tu_gaps, std_907_wt_gaps],
    'mean_filled': [mean_901_am_filled, mean_901_co_filled, mean_901_do_filled, mean_901_ph_filled, mean_901_tu_filled, mean_901_wt_filled,
                mean_905_am_filled, mean_905_co_filled, mean_905_do_filled, mean_905_ph_filled, mean_905_tu_filled, mean_905_wt_filled,
                mean_907_am_filled, mean_907_co_filled, mean_907_do_filled, mean_907_ph_filled, mean_907_tu_filled, mean_907_wt_filled],
    'std_filled': [std_901_am_filled, std_901_co_filled, std_901_do_filled, std_901_ph_filled, std_901_tu_filled, std_901_wt_filled,
                std_905_am_filled, std_905_co_filled, std_905_do_filled, std_905_ph_filled, std_905_tu_filled, std_905_wt_filled,
                std_907_am_filled, std_907_co_filled, std_907_do_filled, std_907_ph_filled, std_907_tu_filled, std_907_wt_filled]
    })

print(results)

#%% Get the percent difference between the mean and standard deviation of the original and filled data
percent_diff_mean_901_am = (mean_901_am_filled - mean_901_am_gaps) / mean_901_am_gaps * 100
percent_diff_mean_901_co = (mean_901_co_filled - mean_901_co_gaps) / mean_901_co_gaps * 100
percent_diff_mean_901_do = (mean_901_do_filled - mean_901_do_gaps) / mean_901_do_gaps * 100
percent_diff_mean_901_ph = (mean_901_ph_filled - mean_901_ph_gaps) / mean_901_ph_gaps * 100
percent_diff_mean_901_tu = (mean_901_tu_filled - mean_901_tu_gaps) / mean_901_tu_gaps * 100
percent_diff_mean_901_wt = (mean_901_wt_filled - mean_901_wt_gaps) / mean_901_wt_gaps * 100

percent_diff_mean_905_am = (mean_905_am_filled - mean_905_am_gaps) / mean_905_am_gaps * 100
percent_diff_mean_905_co = (mean_905_co_filled - mean_905_co_gaps) / mean_905_co_gaps * 100
percent_diff_mean_905_do = (mean_905_do_filled - mean_905_do_gaps) / mean_905_do_gaps * 100
percent_diff_mean_905_ph = (mean_905_ph_filled - mean_905_ph_gaps) / mean_905_ph_gaps * 100
percent_diff_mean_905_tu = (mean_905_tu_filled - mean_905_tu_gaps) / mean_905_tu_gaps * 100
percent_diff_mean_905_wt = (mean_905_wt_filled - mean_905_wt_gaps) / mean_905_wt_gaps * 100

percent_diff_mean_907_am = (mean_907_am_filled - mean_907_am_gaps) / mean_907_am_gaps * 100
percent_diff_mean_907_co = (mean_907_co_filled - mean_907_co_gaps) / mean_907_co_gaps * 100
percent_diff_mean_907_do = (mean_907_do_filled - mean_907_do_gaps) / mean_907_do_gaps * 100
percent_diff_mean_907_ph = (mean_907_ph_filled - mean_907_ph_gaps) / mean_907_ph_gaps * 100
percent_diff_mean_907_tu = (mean_907_tu_filled - mean_907_tu_gaps) / mean_907_tu_gaps * 100
percent_diff_mean_907_wt = (mean_907_wt_filled - mean_907_wt_gaps) / mean_907_wt_gaps * 100

# Store the results in a DataFrame
percent_diff_mean = pd.DataFrame({
    'station': [901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'percent_diff_mean': [percent_diff_mean_901_am, percent_diff_mean_901_co, percent_diff_mean_901_do, percent_diff_mean_901_ph, percent_diff_mean_901_tu, percent_diff_mean_901_wt,
                        percent_diff_mean_905_am, percent_diff_mean_905_co, percent_diff_mean_905_do, percent_diff_mean_905_ph, percent_diff_mean_905_tu, percent_diff_mean_905_wt,
                        percent_diff_mean_907_am, percent_diff_mean_907_co, percent_diff_mean_907_do, percent_diff_mean_907_ph, percent_diff_mean_907_tu, percent_diff_mean_907_wt]
    })

print(percent_diff_mean)

#%% Test if the original and filled data come from the same distribution using the Kolmogorov-Smirnov test
stat_901_am, p_901_am = ks_2samp(am_901_gaps[:500], am_901_filled[:500])
stat_901_co, p_901_co = ks_2samp(co_901_gaps, co_901_filled)
stat_901_do, p_901_do = ks_2samp(do_901_gaps, do_901_filled)
stat_901_ph, p_901_ph = ks_2samp(ph_901_gaps, ph_901_filled)
stat_901_tu, p_901_tu = ks_2samp(tu_901_gaps, tu_901_filled)
stat_901_wt, p_901_wt = ks_2samp(wt_901_gaps, wt_901_filled)

stat_905_am, p_905_am = ks_2samp(am_905_gaps, am_905_filled)
stat_905_co, p_905_co = ks_2samp(co_905_gaps, co_905_filled)
stat_905_do, p_905_do = ks_2samp(do_905_gaps, do_905_filled)
stat_905_ph, p_905_ph = ks_2samp(ph_905_gaps, ph_905_filled)
stat_905_tu, p_905_tu = ks_2samp(tu_905_gaps, tu_905_filled)
stat_905_wt, p_905_wt = ks_2samp(wt_905_gaps, wt_905_filled)

stat_907_am, p_907_am = ks_2samp(am_907_gaps, am_907_filled)
stat_907_co, p_907_co = ks_2samp(co_907_gaps, co_907_filled)
stat_907_do, p_907_do = ks_2samp(do_907_gaps, do_907_filled)
stat_907_ph, p_907_ph = ks_2samp(ph_907_gaps, ph_907_filled)
stat_907_tu, p_907_tu = ks_2samp(tu_907_gaps, tu_907_filled)
stat_907_wt, p_907_wt = ks_2samp(wt_907_gaps, wt_907_filled)

# Store the results in a DataFrame
results = pd.DataFrame({
    'station': [901, 901, 901, 901, 901, 901, 905, 905, 905, 905, 905, 905, 907, 907, 907, 907, 907, 907],
    'variable': ['Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature',
                'Ammonium', 'Conductivity', 'Dissolved oxygen', 'pH', 'Turbidity', 'Water temperature'],
    'statistic': [stat_901_am, stat_901_co, stat_901_do, stat_901_ph, stat_901_tu, stat_901_wt,
                stat_905_am, stat_905_co, stat_905_do, stat_905_ph, stat_905_tu, stat_905_wt,
                stat_907_am, stat_907_co, stat_907_do, stat_907_ph, stat_907_tu, stat_907_wt],
    'p-value': [p_901_am, p_901_co, p_901_do, p_901_ph, p_901_tu, p_901_wt,
                p_905_am, p_905_co, p_905_do, p_905_ph, p_905_tu, p_905_wt,
                p_907_am, p_907_co, p_907_do, p_907_ph, p_907_tu, p_907_wt]
})

print(results)

# # NOTE: The p-value is the probability of observing the given statistic if the null hypothesis is true. If the p-value is less than the significance level (0.05), we reject the null hypothesis. In this case, the null hypothesis is that the two samples come from the same distribution. If the p-value is greater than 0.05, we fail to reject the null hypothesis.
# # In other words, if p-value < 0.05, we can conclude that the original and filled data come from different distributions. If p-value > 0.05, we can conclude that the original and filled data come from the same distribution.
