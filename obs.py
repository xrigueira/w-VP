"""Contains obsolete methods just in case"""

# Original windower. Not multiresolution
# def windower(self, data):

#     """Takes a 2D array with multivariate time series data
#     and creates sliding windows. The arrays store the different
#     variables in a consecutive manner. E.g. [first 6 variables,
#     next 6 variables, and so on].
#     ----------
#     Arguments:
#     data (pickle): file with the time-series data to turn 
#     into windows.
#     num_variables (int): the number of variables in the data.
#     window_size (int): the size of the windows.
#     stride (int): the stride of the windows.
    
#     Returns:
#     windows (np.array): time series data grouped in windows."""
    
#     windows = []
#     for i in data:
        
#         # Get the number of windows
#         num_windows = (len(i) - self.window_size * self.num_variables) // self.num_variables + 1
        
#         # Create windows
#         for j in range(0, num_windows, self.stride * self.num_variables):
#             window = i[j:j+self.window_size * self.num_variables]
#             windows.append(window)

#     # Convert the result to a NumPy array
#     windows = np.array(windows)
    
#     return windows