import pickle
import matplotlib.pyplot as plt

from dater import dater

def plotter(data, num_variables, windowed):
    
    """This function plots the original data,
    and the windowed data at all resolution levels.
    ---------
    Arguments:
    data: The data to be plotted.
    num_variables: The number of variables in the data.
    windowed: Whether the data is windowed or not.
    
    Returns:
    None"""

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

if __name__ == '__main__':

    # Read the last anomaly data file
    file_anomalies = open(f'pickels/anomaly_data_test.pkl', 'rb')
    anomalies_windows = pickle.load(file_anomalies)
    file_anomalies.close()

    # Rename data
    X = anomalies_windows

    stride = 1
    num_variables = 6

    num_windows_high = [i - 32 + 1 for i in X[-1]]
    print(num_windows_high)
    
    # print(dater(901, X[0][0][14]))
        # print(dater(X[0][i], num_variables, stride, num_windows_high[i], windowed=True))
    # Plot the data
    # plotter(data_low, num_variables=6, windowed=True)