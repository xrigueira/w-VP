import pickle
import matplotlib.pyplot as plt

"""Plot the multivariate data of those windows that are anomalous and background respectively"""

# Read the windowed anomalous data
file_anomalies = open('anomaly_data_1.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Read the windowed background data
file_background = open('background_data_0.pkl', 'rb')
background_windows = pickle.load(file_background)
file_background.close()

anomalies_windows = anomalies_windows.reshape(-1, 6)

grouped_data = [anomalies_windows[i:i+16] for i in range(0, len(anomalies_windows), 16)]
print(grouped_data[-2])

# Number of rows in each group
num_data_points = 16

# Loop through the groups and create separate plots for each
for group in grouped_data:
    
    # Create X-axis values representing the indices of data points
    x_values = list(range(1, num_data_points + 1))

    # Create a line plot for each variable
    for i in range(6):
        y_values = [row[i] for row in group]
        plt.plot(x_values, y_values, label=f'Variable {i+1}')

    # Set labels and legend
    plt.xlabel('Data Point Index')
    plt.ylabel('Variable Value')
    plt.legend()

    # Show the plot
    plt.show()