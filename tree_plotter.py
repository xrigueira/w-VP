import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree

# Load the previous Random Forest models
filename = f'models/rf_model_high_{1}.sav'
loaded_model_high = pickle.load(open(filename, 'rb'))
filename = f'models/rf_model_med_{1}.sav'
loaded_model_med = pickle.load(open(filename, 'rb'))
filename = f'models/rf_model_low_{1}.sav'
loaded_model_low = pickle.load(open(filename, 'rb'))

# Plot an estimator (tree) of the Random Forest model
from sklearn.tree import export_graphviz

# Define feature names (for a med tree in this case)
feature_names = ['am-8', 'co-8', 'do-8', 'ph-8', 'wt-8', 'tu-8',
                'am-7', 'co-7', 'do-7', 'ph-7', 'wt-7', 'tu-7',
                'am-6', 'co-6', 'do-6', 'ph-6', 'wt-6', 'tu-6',
                'am-5', 'co-5', 'do-5', 'ph-5', 'wt-5', 'tu-5',
                'am-4', 'co-4', 'do-4', 'ph-4', 'wt-4', 'tu-4',
                'am-3', 'co-3', 'do-3', 'ph-3', 'wt-3', 'tu-3',
                'am-2', 'co-2', 'do-2', 'ph-2', 'wt-2', 'tu-2',
                'am-1', 'co-1', 'do-1', 'ph-1', 'wt-1', 'tu-1',
                'am+0', 'co+0', 'do+0', 'ph+0', 'wt+0', 'tu+0',
                'am+1', 'co+1', 'do+1', 'ph+1', 'wt+1', 'tu+1',
                'am+2', 'co+2', 'do+2', 'ph+2', 'wt+2', 'tu+2',
                'am+3', 'co+3', 'do+3', 'ph+3', 'wt+3', 'tu+3',
                'am+4', 'co+4', 'do+4', 'ph+4', 'wt+4', 'tu+4',
                'am+5', 'co+5', 'do+5', 'ph+5', 'wt+5', 'tu+5',
                'am+6', 'co+6', 'do+6', 'ph+6', 'wt+6', 'tu+6',
                'am+7', 'co+7', 'do+7', 'ph+7', 'wt+7', 'tu+7']

# Plot the tree
export_graphviz(loaded_model_med.estimators_[0], out_file='tree.dot',
                feature_names=feature_names, # Number of data points in a window
                rounded=True,
                proportion=False,
                precision=2,
                filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_0.png', '-Gdpi=600'])
