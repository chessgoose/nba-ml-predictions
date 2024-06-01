import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_data
from sklearn.model_selection import GridSearchCV
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the hyperparameter grid
x_train, y_train = load_data("wnba")
#x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.075],
    'subsample': [0.8, 1],
    'objective': ['binary:logistic']
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the GridSearchCV object
# n_splits small because my shit small
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', verbose=2)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)