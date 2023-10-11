import random
import pickle
import numpy as np
import pandas as pd

"""This file contains the main class imRF which implements
iterative multiresolution Random Forest."""

class imRF():
    
    def __init__(self, station, trim_percentage, ratio, seed) -> None:
        
        self.station = station
        self.trim_percentage = trim_percentage
        self.ratio = ratio
        self.seed = seed
    
    def anomalies(self):
        
        """Extracts the anomalies from the database and
        saves them to a pickle file.
        ----------
        Arguments:
        self.
        
        Stores:
        anomaly_data (pickle): file with the multivariate data
        from each anomaly.
        
        Returns:
        trimmed_anomalies_indexes (list): start and end index of each anomaly.
        """
        
        # Load the data
        data = pd.read_csv(f'data/labeled_{self.station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 1
        data_anomalies = data[data["label"] == 1]
        
        # Create a new column with the difference between consecutive dates
        date_diff = (data_anomalies['date'] - data_anomalies['date'].shift()).fillna(pd.Timedelta(minutes=15))

        # Create groups of consecutive dates
        date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

        # Get the starting and ending indexes of each group of consecutive dates
        grouped = data.groupby(date_group)
        consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]
        
        # Trim the start and end of the anomalies to remove the onset and the offset
        trimmed_anomalies_indexes = []
        for start, end in consecutive_dates_indexes:
            anomaly_length = end - start
            trim_amount = int(anomaly_length * self.trim_percentage / 100)
            trimmed_start = start + trim_amount
            trimmed_end = end - trim_amount
            trimmed_anomalies_indexes.append((trimmed_start, trimmed_end))
        
        # Extract the data
        anomaly_data = []
        for start, end in trimmed_anomalies_indexes:
            subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
            anomaly_data.append(subset_rows)
        
        # Save anomalys_data to disk as pickle object
        with open('anomaly_data.pkl', 'wb') as file:
            pickle.dump(anomaly_data, file)
        
        return trimmed_anomalies_indexes
    
    def init_background(self, anomalies_indexes):
        
        """Creates the initial background file by extracting
        'ratio' times more non anomalous data than the anomaly method.
        The data is saved to a pickle file.
        -----------
        Arguments:
        self.
        
        Saves:
        background_data (pickle): file with 'ratio' times more 
        nonanomalous data, also know as background, compared to the
        total legth of the anomalies in the dataset.
        
        Returns:
        background_indexes (list): start and end indexes of the extracted
        background data.
        """
        
        # Define random seed
        random.seed(self.seed)
        
        # Load the DataFrame from your dataset
        data = pd.read_csv(f'data/labeled_{self.station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 0
        data_background = data[data["label"] == 0]
        
        # Filter the dataset to include only days that meet the ammonium level the condition
        mean_ammonium = np.mean(data_background.ammonium_901)
        data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x['ammonium_901'].max() <= mean_ammonium)
        
        # Extract the length of the anomalies
        len_anomalies = [end - start for start, end in anomalies_indexes]
        
        # Define background data indexes
        background_indexes = []
        for anomaly_length in len_anomalies:
            if anomaly_length != 0:
                start = random.randint(0, len(data_background) - 1)
                end = start + (anomaly_length * self.ratio)
                background_indexes.append((start, end))
        
        # Extract the data
        background_data = []
        for start, end in background_indexes:
            
            subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
            background_data.append(subset_rows)
        
        # Save anomalies_data to disk as numpy object
        with open('background_data.pkl', 'wb') as file:
            pickle.dump(background_data, file)
            
        return background_indexes

if __name__ == '__main__':
    
    # Create an instance of the model
    imRF = imRF(station=901, trim_percentage=10, ratio=5, seed=0)
    
    # Implement iterative process
    for i in range(0, 5):
        
        if i == 0:
            # Extract the anomalies and first batch of background
            anomalies_indexes = imRF.anomalies()
            
            background_indexes = imRF.init_background(anomalies_indexes)

        else:
            pass
            # Here would go the rest of the process