import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

"""This file contains the main class imRF which implements
iterative multiresolution Random Forest."""

class imRF():
    
    def __init__(self, station, trim_percentage, ratio, num_variables, window_size, stride, seed) -> None:
        
        self.station = station
        self.trim_percentage = trim_percentage
        self.ratio = ratio
        self.num_variables = num_variables
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
    
    def windower(self, data):
        
        """
        Takes a 2D array with multivariate time series data
        and creates multiresolution sliding windows. The size
        of the slinding windows gets halved each time. The
        resulting windows store different variables in a 
        consecutive manner. E.g. [first 6 variables, next 6 
        variables, and so on].
        ----------
        Arguments:
        data (pickle): file with the time-series data to turn 
        into windows.
        num_variables (int): the number of variables in the data.
        window_size (int): the size of the windows.
        stride (int): the stride of the windows.
        
        Returns:
        windows (list): time series data grouped in windows"""
        
        windows = []
        if self.window_size > 4: # This way the maximum window would be 8 data points
            
            for i in data:
                
                # Get the number of windows
                num_windows = (len(i) - self.window_size * self.num_variables) // (self.stride * self.num_variables) + 1
                
                # Create the windows
                for j in range(0, num_windows, self.stride):
                    window = i[j * self.num_variables: (j * self.num_variables) + (self.window_size * self.num_variables)]
                    windows.append(window)
            
            self.window_size = self.window_size // 2 # Halve window size
            
            return [windows] + self.windower(data)  # Recursive call
        
        else:
            
            # Restore the window size after recursion
            self.window_size = window_size
            return []
    
    def anomalies(self):
        
        """Extracts the anomalies from the database and
        saves them to a pickle file.
        ----------
        Arguments:
        self.
        
        Stores:
        anomaly_data (pickle): file with the multivariate data.
        from each anomaly.
        
        Returns:
        trimmed_anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        """
        
        # Load the data
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
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

        # Group the data in windows before saving
        anomaly_data = self.windower(anomaly_data)
        
        # Save anomaly_data to disk as pickle object
        with open('pickels/anomaly_data_0.pkl', 'wb') as file:
            pickle.dump(anomaly_data, file)
        
        return trimmed_anomalies_indexes
    
    def init_background(self, anomalies_indexes):
        
        """Creates the initial background file by extracting
        'ratio' times more non anomalous data than the anomaly method.
        The data is saved to a pickle file.
        -----------
        Arguments:
        self.
        anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        
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
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 0
        data_background = data[data["label"] == 0]
        
        # Filter the dataset to include only days that meet the ammonium level the condition
        mean_ammonium = np.mean(data_background.ammonium_901)
        data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x[f'ammonium_{self.station}'].max() <= mean_ammonium)
        
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
        
        # Group data into windows before saving
        background_data = self.windower(background_data)
        
        # Save anomalies_data to disk as numpy object
        with open(f'pickels/background_data_0.pkl', 'wb') as file:
            pickle.dump(background_data, file)
            
        return background_indexes
    
    def background(self, anomalies_indexes, background_indexes, iteration):
        
        """Creates the background file for each iteration by extracting
        'ratio' times more non anomalous data than the anomaly method and
        adding that that to the previous background file. The data is saved
        to a pickle file. It makes sure that the new non anomalous data 
        extracted has not been selected before.
        ----------
        Arguments:
        self.
        anomalies_indexes (list): start and end indexes of the extracted
        anomaly data.
        background_indexes (list): start and end indexes of the previously
        extracted background data.
        iteration (int): the current iteration number.
        
        Saves:
        background_data_i (pickle): file with 'ratio' times more 
        nonanomalous data, compared to the total legth of the 
        anomalies in the dataset.
        
        Returns:
        background_indexes (list): updated start and end indexes of the 
        extracted background data.
        """
        
        # Define random seed
        random.seed(self.seed)
    
        # Load the DataFrame from your dataset
        data = pd.read_csv(f'data/labeled_{self.station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])
        
        # Filter the data to select only rows where the label column has a value of 0
        data_background = data[data["label"] == 0]
        
        # Filter the dataset to include only days that meet the ammonium level the condition
        mean_ammonium = np.mean(data_background.ammonium_901)
        data_background = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x[f'ammonium_{self.station}'].max() <= mean_ammonium)
        
        # Extract the length of the anomalies
        len_anomalies = [end - start for start, end in anomalies_indexes]

        # Define new background data indexes
        new_background_indexes = []
        for anomaly_length in len_anomalies:
            if anomaly_length != 0:
                new_start = random.randint(0, len(data_background) - 1)
                new_end = new_start + (anomaly_length * self.ratio)
                
                # Check for overlap
                overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
                
                # If there is an overlap, generate a new index
                max_retries = 10  # Set a maximum number of retries
                retry_count = 0
                while overlaps:
                    new_start = random.randint(0, len(data_background) - 1)
                    new_end = new_start + (anomaly_length * self.ratio)
                    overlaps = any(start <= new_end and end >= new_start for start, end in background_indexes)
                    retry_count += 1
                
                    if retry_count == max_retries:
                        break
                
                # Append the nonoverlaping indexes to the new list and the old one
                new_background_indexes.append((new_start, new_end))
                background_indexes.append((new_start, new_end))
        
        # Extract the data
        background_data = []
        for start, end in new_background_indexes:
            
            subset_rows = data_background.iloc[start:end + 1, 1:-2].values.flatten() # Extarct rows withing the subset
            background_data.append(subset_rows)
        
        # Group data into windows before saving
        background_data = self.windower(background_data)
        
        # Save anomalies_data to disk as pickle object
        with open(f'pickels/background_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(background_data, file)
            
        return background_indexes
    
    def init_RandomForest(self):
        
        """Initiates a Random Forest classifier in the first
        iteration. The model gets trained on the first batch
        of anomalies and background data extrated previously.
        ----------
        Arguments:
        self.
        
        Saves:
        rf_model_0 (sav): file with the model saved to disk.
        
        Returns:
        """
        
        # Read the windowed anomalous data
        file_anomalies = open('pickels/anomaly_data_0.pkl', 'rb')
        anomalies_windows = pickle.load(file_anomalies)
        file_anomalies.close()

        # Read the windowed background data
        file_background = open('pickels/background_data_0.pkl', 'rb')
        background_windows = pickle.load(file_background)
        file_background.close()
        
        # Generate labels for each window
        anomalies_labels = []
        for i in range(len(anomalies_windows)):
            anomalies_labels.append(np.array([1 for j in anomalies_windows[i]]))
        
        background_labels = []
        for i in range(len(background_windows)):
            background_labels.append(np.array([0 for j in background_windows[i]]))
        
        # Concatenate array
        X = []
        for i in range(len(anomalies_windows)):
            X.append(np.concatenate((anomalies_windows[i], background_windows[i])))
        
        # Continue debugging here. Probably something fishy going on with the randomizer
        # y = []
        # for i in range(len(anomalies_windows)):
        #     y.append(np.concatenate((anomalies_labels[i], background_labels[i])))
        
        # # Shuffle data
        # randomized = []
        # for i in range(len(anomalies_windows)):
        #     combined = np.column_stack((X[i], y[i]))
        #     np.random.seed(self.seed)
        #     np.random.shuffle(combined)
        #     randomized.append(combined)

        # # Split the shuffled array back into data and labels
        # for i in range(len(anomalies_windows)):    
        #     X[i], y[i] = randomized[i][:, :-1], randomized[i][:, -1]
        
        # # Train the Random Forest classifiers
        # model_high = RandomForestClassifier(random_state=self.seed)
        # model_med = RandomForestClassifier(random_state=self.seed)
        # model_low = RandomForestClassifier(random_state=self.seed)
        
        # # Split the shuffled data into the training and testing set
        # X_train, y_train, X_test, y_test = [], [], [], []
        # for i in range(len(anomalies_windows)):
        #     X_train.append(X[i][:int(len(X) * 0.75)])
        #     y_train.append(y[i][:int(len(X) * 0.75)])
        #     X_test.append(X[i][int(len(X) * 0.75):])
        #     y_test.append(y[i][int(len(X) * 0.75):])

        # # Fit the model to the training data
        # model_high.fit(X_train[0], y_train[0]) # Long length data windows
        # model_med.fit(X_train[1], y_train[1]) # Medium legth data windows
        # model_low.fit(X_train[2], y_train[2]) # Short length data windows

        # from sklearn.metrics import confusion_matrix as cm
        # confusion_matrix_high = cm(y_test[0], model_high.predict(X_test[0]))
        # print(confusion_matrix_high)
        # confusion_matrix_med = cm(y_test[1], model_med.predict(X_test[1]))
        # print(confusion_matrix_med)
        # confusion_matrix_low = cm(y_test[2], model_low.predict(X_test[2]))
        # print(confusion_matrix_low)
        
        # # Get the number of rows labeled as anomalies in y_test
        # num_anomalies = len([i for i in y_test[0] if i==1])
        # print('Number of anomalies in test set:', num_anomalies)
        # num_anomalies = len([i for i in y_test[1] if i==1])
        # print('Number of anomalies in test set:', num_anomalies)
        # num_anomalies = len([i for i in y_test[2] if i==1])
        # print('Number of anomalies in test set:', num_anomalies)
        
        # # Save the model to disk
        # filename = 'models/rf_model_0.sav'
        # pickle.dump(model, open(filename, 'wb'))
        
    def RandomForest(self, iteration):
        
        """Updates the Random Forest model on each iterations.
        The older model performs prediction on new background data.
        Those windows classified as anomalies get added to previous
        anomaly data and those which are background get included in the
        previous background data. The older model gets retrained and
        saved as new version.
        ----------
        Arguments:
        self.
        iteration (int): the current iteration number.
        
        Saves:
        rf_model_{iteration} (sav): with the updated model saved to disk.
        
        Returns:
        """
        
        # Read the current windowed background
        file_background = open(f'pickels/background_data_{iteration}.pkl', 'rb')
        background_windows = pickle.load(file_background)
        file_background.close()
        
        # Variable name change to follow best practives in ML
        X = background_windows

        # Shuffle the data
        np.random.seed(self.seed)
        np.random.shuffle(X)

        # Load the previous model
        filename = f'models/rf_model_{iteration - 1}.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        
        # Get the results from each tree
        trees = loaded_model.estimators_

        tree_classifications = [tree.predict(X) for tree in trees]

        # Get the average score for each windows
        score_Xs = np.mean(tree_classifications, axis=0)

        plt.plot(score_Xs)
        # plt.show()
        plt.savefig(f'images/prediction_{iteration}.png', dpi=300)
        
        # Get the indexes of those windows that are anomalies and background in the new data
        indexes_anomalies_windows = list(np.where(score_Xs >= 0.51)[0])
        indexes_background_windows = list(np.where(score_Xs <= 0.10)[0])

        # Extract those new anomaly and background windows
        add_anomalies_windows = background_windows[indexes_anomalies_windows]
        print(f'Percentage of anomalies {round(len(add_anomalies_windows) / len(background_windows) * 100, 2)}%')
        add_background_windows = background_windows[indexes_background_windows]

        # Read the previous windowed anomalous data
        file_anomalies = open(f'pickels/anomaly_data_{iteration - 1}.pkl', 'rb')
        prev_anomalies_windows = pickle.load(file_anomalies)
        file_anomalies.close()

        # Read the previous windows background
        file_background = open(f'pickels/background_data_{iteration - 1}.pkl', 'rb')
        prev_background_windows = pickle.load(file_background)
        file_background.close()
        
        # Conactenate new data with old data
        anomalies_windows = np.vstack((prev_anomalies_windows, add_anomalies_windows))
        background_windows = np.vstack((prev_background_windows, add_background_windows))

        # Save anomalies_data to disk as pickle object
        with open(f'pickels/anomaly_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(anomalies_windows, file)
        
        # Save background data as a pickle object
        with open(f'pickels/background_data_{iteration}.pkl', 'wb') as file:
            pickle.dump(background_windows, file)
        
        # Retrain the model with the updated anomaly and background data
        # Generate labels for each window
        anomalies_labels = np.array([1 for i in anomalies_windows])
        background_labels = np.array([0 for i in background_windows])
        
        # Concatenate arrays
        X = np.concatenate((anomalies_windows, background_windows))
        y = np.concatenate((anomalies_labels, background_labels))

        # Shuffle data
        combined = np.column_stack((X, y))
        np.random.seed(self.seed)
        np.random.shuffle(combined)

        # Split the shuffled array back into data and labels
        X, y = combined[:, :-1], combined[:, -1]

        # Load the model
        filename = f'models/rf_model_{iteration - 1}.sav'
        model = pickle.load(open(filename, 'rb'))

        # Increase estimators and set warm_start to True
        model.n_estimators += 10
        model.warm_start = True

        # Split the shuffled data into the training and testing set
        X_train, y_train = X[:int(len(X) * 0.75)], y[:int(len(X) * 0.75)]
        X_test, y_test = X[int(len(X) * 0.75):], y[int(len(X) * 0.75):]

        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        from sklearn.metrics import confusion_matrix as cm
        confusion_matrix = cm(y_test, model.predict(X_test))
        print(confusion_matrix)

        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in y_test if i==1])
        print('Number of anomalies in test set:', num_anomalies)

        # Save the model to disk
        filename = f'models/rf_model_{iteration}.sav'
        pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    
    # Create an instance of the model
    window_size = 32
    imRF = imRF(station=901, trim_percentage=10, ratio=5, num_variables=6, 
                window_size=window_size, stride=1, seed=0)
    
    # Implement iterative process
    for i in range(0, 10):
        
        if i == 0:
            print(f'[INFO] Iteration {i}')
            # Extract the anomalies and first batch of background
            anomalies_indexes = imRF.anomalies()
            
            background_indexes = imRF.init_background(anomalies_indexes)
            
            # Train the first version of the model
            imRF.init_RandomForest()

        else:
            break
            print(f'[INFO] Iteration {i}')
            # Extract new background data
            background_indexes = imRF.background(anomalies_indexes, background_indexes, iteration=i)
            
            # # Iteratively predict on the new background data and update the model
            # imRF.RandomForest(iteration=i)
