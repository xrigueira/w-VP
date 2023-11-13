import pickle
import pandas as pd
import numpy as np

# Load the previous Random Forest models
filename = f'models/rf_model_high_{1}.sav'
loaded_model_high = pickle.load(open(filename, 'rb'))
filename = f'models/rf_model_med_{1}.sav'
loaded_model_med = pickle.load(open(filename, 'rb'))
filename = f'models/rf_model_low_{1}.sav'
loaded_model_low = pickle.load(open(filename, 'rb'))

# Plot an estimator (tree) of the Random Forest model
from sklearn.tree import export_graphviz
export_graphviz(loaded_model_med.estimators_[30], out_file='tree.dot',
                feature_names=range(96), # Number of data points in a window
                rounded=True,
                proportion=False,
                precision=2,
                filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_30.png', '-Gdpi=600'])
