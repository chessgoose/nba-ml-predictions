import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data, drop_regression_stats
import warnings
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
https://github.com/slavafive/ML-medium/blob/master/quantile_regression.ipynb
"""
league = "nba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 42
np.random.seed(random_seed)

data = load_regression_data('wnba')
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

drop_regression_stats(data)
print(data.head())

# FIT ON DIFFERENCE between POINTS and DARKO, LOG TRANSFORM EVERYTHING?
relative_performance = data['Relative Performance']
data['Difference'] = data['Points'] - data['DARKO']

shifted_relative_performance = relative_performance - relative_performance.min() + 1  # Shift to make all values positive
sqrt_relative_performance = np.sqrt(shifted_relative_performance)

y_axis = data['Difference']

acc_results = []
x_train, x_test, y_train, y_test = train_test_split(np.array(sqrt_relative_performance).reshape(-1, 1), y_axis, test_size=0.2, shuffle=True, random_state=random_seed)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict using the test set
predictions = model.predict(x_test)

# Print MAE 
mae = mean_absolute_error(y_test, predictions)
print(f'MAE: {mae}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Predictions')

# Plot a line of perfect prediction for reference
min_val = min(min(y_test), min(predictions))
max_val = max(max(y_test), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

# Add labels and title
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# Optionally, you can calculate and store accuracy metrics
print(model.score(x_test, y_test))

import pickle

# Save the model to a file
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)