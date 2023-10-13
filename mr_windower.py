import numpy as np

data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]

def windower(data, window_size, stride, num_variables):

    windows = []
    if window_size > 1:
        for i in data:
            num_windows = (len(i) - window_size * num_variables) // (stride * num_variables) + 1
            for j in range(0, num_windows, stride):
                window = i[j * num_variables: (j * num_variables) + (window_size * num_variables)]
                windows.append(window)
        window_size = window_size // 2
        return [windows] + windower(data, window_size, stride, num_variables)
    else:
        return []

window_size = 4
num_variables = 1
stride = 1

windows = windower(data, window_size, stride, num_variables)

print(windows)