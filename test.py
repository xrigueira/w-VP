import matplotlib.pyplot as plt

data = [
    [0.92682927, 0.62605549, 0.34444444, 0.70754717, 0.08695652, 0.60891089],
    [0.92682927, 0.63208685, 0.33333333, 0.70754717, 0.08695652, 0.60891089],
    [0.92682927, 0.63329312, 0.33333333, 0.70754717, 0.08695652, 0.60891089]
]

# Calculate the number of data points
num_data_points = len(data)

# Create X-axis values representing the indices of data points
x_values = list(range(1, num_data_points + 1))

# Create a line plot for each variable
for i in range(6):
    y_values = [row[i] for row in data]
    plt.plot(x_values, y_values, label=f'Variable {i+1}')

# Set labels and legend
plt.xlabel('Data Point Index')
plt.ylabel('Variable Value')
plt.legend()

# Show the plot
plt.show()