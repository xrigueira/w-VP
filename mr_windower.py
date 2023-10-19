import numpy as np

data = [[*range(1, 65, 1)]]

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

window_size = 8
num_variables = 6
stride = 1

windows = windower(data, window_size, stride, num_variables)

# print(windows)
