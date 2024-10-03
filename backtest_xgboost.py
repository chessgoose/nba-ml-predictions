
# Try Kelly strategy + fixed bet size
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from get_odds import get_odds_today, get_matchups, get_team_points_today
from utils.dataloading import drop_regression_stats, load_regression_data
from wnba_reg import calculate_wnba_features
from utils.odds import calculate_kelly_criterion
from train_nn_regressor import QuantileRegressor
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)
ou_model_name = 'models/regression/wnba/XGBoost_2023.json'
model = xgb.Booster()
model.load_model(ou_model_name) 

data = load_regression_data(league)
z_test = data['Line']
OU = data['OU Result']
y_test = data['Points']

drop_regression_stats(data)
data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)

variances = data['Variance'].values
data.drop(['Variance'], axis=1, inplace=True)
print(data)

predictions = model.predict(xgb.DMatrix(data, missing=-np.inf))

y_lower = predictions[:, 0]  # alpha=0.476
y_upper = predictions[:, 1]  # alpha=0.524

padding = 0
variance_threshold = 30
valid_indices = np.where((z_test < np.minimum(y_upper, y_lower) - padding) | (z_test > np.maximum(y_lower, y_upper) + padding))[0]
#valid_indices = valid_indices[np.where(variances[valid_indices] < variance_threshold)]

valid_predictions = (y_lower[valid_indices] + y_upper[valid_indices]) / 2
valid_y_test = y_test.iloc[valid_indices]
valid_z_test = z_test.iloc[valid_indices]

mae = mean_absolute_error(valid_y_test, valid_predictions)
print(f"MAE: {mae}")

# Compare the predictions to the lines to compute accuracy when compared to over/under
predicted_ou_results = np.where(valid_predictions > valid_z_test, 1, 0)
actual_ou_results = np.where(valid_y_test > valid_z_test, 1, 0)
acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
print(f"Accuracy: {acc}% on {len(predicted_ou_results)} results")

plt.figure(figsize=(10, 6))
plt.scatter(valid_y_test, valid_predictions, color='blue', label='Predictions')

# Plot a line of perfect prediction for reference
min_val = min(min(valid_y_test), min(valid_predictions))
max_val = max(max(valid_y_test), max(valid_predictions))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

# Add labels and title
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()

# Show the plot
plt.show()

