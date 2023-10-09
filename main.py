import pickle

# Read the data
file_anomalies = open('anomaly_data.pkl', 'rb')
anomalies = pickle.load(file_anomalies)
file_anomalies.close()

file_background = open('background_data.pkl', 'rb')
background = pickle.load(file_background)
file_background.close()

# Train a classifier?
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)

# Fit the model to the training data
