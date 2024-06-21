import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data
import warnings
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
https://github.com/slavafive/ML-medium/blob/master/quantile_regression.ipynb
"""
league = "nba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 1234
np.random.seed(random_seed)

data = load_regression_data()
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

data.drop(["OU Result", "Line", "Points", "FG T", "Minutes Diff"], axis=1, inplace=True)
print(data.head())

acc_results = []
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=0.2, shuffle=True, random_state=random_seed)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict using the test set
predictions = model.predict(x_train)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, predictions, color='blue', label='Predictions')

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
acc_results.append(model.score(x_test, y_test))
print(model.score(x_test, y_test))