data = {'high': {'am-1': [10, 3, 1, 0],
                'am+1': [20, 4, 0, 3],
                'co+1': [70, 9, 0, 3]},
        'med': {'am-1': [30, 5, 1, 0],
                'am+1': [40, 6, 0, 3],
                'co+1': [80, 10, 0, 3]},
        'low': {'am-1': [50, 7, 1, 0],
                'am+1': [60, 8, 0, 3],
                'co+1': [90, 11, 0, 3]}}

# Step 1: Combine all 'am' values and 'co' values
am_values = []
co_values = []

for category in data.values():
    am_values.extend(category['am-1'])
    am_values.extend(category['am+1'])
    co_values.extend(category['co+1'])

# Step 2: Calculate min and max for 'am' and 'co'
am_min, am_max = min(am_values), max(am_values)
co_min, co_max = min(co_values), max(co_values)

# Step 3: Normalize each value
normalized_data = {}
for category, values in data.items():
    normalized_data[category] = {}
    normalized_data[category]['am-1'] = [(val - am_min) / (am_max - am_min) for val in values['am-1']]
    normalized_data[category]['am+1'] = [(val - am_min) / (am_max - am_min) for val in values['am+1']]
    normalized_data[category]['co+1'] = [(val - co_min) / (co_max - co_min) for val in values['co+1']]

# Step 4: Print normalized data
print(normalized_data)