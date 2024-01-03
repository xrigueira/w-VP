import pickle
import matplotlib.pyplot as plt

def plotter(data, num_variables, windowed):
    
    if windowed == False:

        data_reshaped = data.reshape(-1, num_variables)

        # Plot each variable
        for i in range(num_variables):
            plt.plot(data_reshaped[:, i], label=f'Variable {i+1}')

        plt.xlabel('Time/Index')
        plt.ylabel('Variable Value')
        plt.legend()
        plt.show()

    if windowed == True:

        for window_index, window in enumerate(data):
            
            # Reshape the window
            window_reshaped = window.reshape(-1, num_variables)

            # Create a new figure for each window
            plt.figure(window_index)

            # Plot each variable
            for i in range(num_variables):
                plt.plot(window_reshaped[:, i], label=f'Variable {i+1}')

            plt.xlabel('Time/Index')
            plt.ylabel('Variable Value')
            plt.legend()
            plt.title(f'Window {window_index+1}')

        plt.show()

# Read the last anomaly data file
file_anomalies = open(f'pickels/anomaly_data_4.pkl', 'rb')
anomalies_windows = pickle.load(file_anomalies)
file_anomalies.close()

# Rename
X = anomalies_windows

stride = 1
num_variables = 6
med_subwindow_span = len(X[1][0]) // (num_variables * stride)
low_subwindow_span = (len(X[0][0])- len(X[2][0])) // (num_variables * stride)

data_high = X[0][0]
data_med = X[1][:(med_subwindow_span + 1)]
data_low = X[2][:(low_subwindow_span + 1)]

# Plot the data
plotter(data_low, num_variables=6, windowed=True)