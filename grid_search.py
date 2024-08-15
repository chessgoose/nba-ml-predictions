import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_data, load_regression_data, load_2023_data
from sklearn.model_selection import GridSearchCV
import warnings
from itertools import combinations
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the hyperparameter grid
#x_train = load_regression_data("wnba")
x_train = load_2023_data()
y_train = x_train["Points"]
x_train.drop(["Points"], axis=1, inplace=True)

#x_train.drop(["Points", "Line", "OU Result"], axis=1, inplace=True)
#x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)
print(x_train)

# Define the hyperparameter grid
"""
param_grid = {
    'max_depth': [3],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'objective': ['binary:logistic']
}
"""

# Load the model from the file
with open('linear_regression_model_new.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

relative_performance = x_train['Relative Performance']

shifted_relative_performance = relative_performance + 35 + 1  # Shift to make all values positive
sqrt_relative_performance = np.log10(shifted_relative_performance)

# Now you can use the loaded model to make predictions
predictions = loaded_model.predict(np.array(sqrt_relative_performance).reshape(-1, 1))

x_train['Adjusted Points'] = x_train['DARKO'] + predictions

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05],
    'subsample': [0.7, 0.8],
    'objective': ['reg:quantileerror'],
    "quantile_alpha": [0.5]
}

# Create the XGBoost model object
xgb_model = xgb.XGBRegressor()

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='neg_mean_absolute_error', verbose=0)

# Get all possible combinations of features
feature_combinations = []
for i in range(1,3):
    feature_combinations.extend(combinations(x_train.columns, i))

# To store the results
results = []

# Iterate through each feature combination
for feature_subset in tqdm(feature_combinations, desc="Feature combinations"):
    x_train_subset = x_train[list(feature_subset)]
    
    # Fit the GridSearchCV object to the training data
    grid_search.fit(x_train_subset, y_train)
    
    # Store the results
    results.append({
        'features': feature_subset,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    })

# Sort the results by best score
results = sorted(results, key=lambda x: x['best_score'], reverse=True)

# Get the top 5 results
top_5_results = results[:10]

# Convert to a DataFrame for saving
top_5_df = pd.DataFrame(top_5_results)

# Save the top 5 results to a CSV file
top_5_df.to_csv('top_5_feature_combinations.csv', index=False)

# Print the top 5 results
print("Top 5 feature combinations:")
print(top_5_df)

