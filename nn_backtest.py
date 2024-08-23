#  Backtesting script for XGBoost 2023 models
# 
#  Convert 2024 dataset into DMatrix
#   Run and compare

# Try Kelly strategy + fixed bet size - Kelly based on difference between line and thing

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
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import joblib

league = "wnba"
data = load_regression_data(league)
z_test = data['Line']
y_test = data['Points']

drop_regression_stats(data)
data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)

variances = data['Variance'].values
data.drop(['Variance'], axis=1, inplace=True)
print(data)

# Initialize the scaler
scaler = joblib.load('nn_models/scaler.pkl')
x_test = scaler.transform(data)

val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

model = QuantileRegressor(2, 2)
model.load_state_dict(torch.load('nn_models/wnba/NN.pth'))
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, _ in val_loader:
        outputs = model(inputs)
        all_preds.append(outputs)

predictions = torch.cat(all_preds).numpy()

y_lower = predictions[:, 0]  # alpha=0.476
y_upper = predictions[:, 1]  # alpha=0.524

padding = 1.0
valid_indices = np.where((z_test < np.minimum(y_upper, y_lower) - padding) | (z_test > np.maximum(y_lower, y_upper) + padding))[0]

if len(valid_indices) == 0:
    print("No valid predictions outside the range [predictions_low, predictions_high]")
else:
    valid_predictions = (y_lower[valid_indices] + y_upper[valid_indices]) / 2
    valid_y_test = y_test.iloc[valid_indices]
    valid_z_test = z_test.iloc[valid_indices]

    mae = mean_absolute_error(valid_y_test, valid_predictions)
    print(f"MAE: {mae}")

    predicted_ou_results = np.where(valid_predictions > valid_z_test, 1, 0)
    actual_ou_results = np.where(valid_y_test > valid_z_test, 1, 0)
    acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
    print(f"Accuracy: {acc}% on {len(predicted_ou_results)} results")

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_y_test, valid_predictions, color='blue', label='Predictions')

    min_val = min(min(valid_y_test), min(valid_predictions))
    max_val = max(max(valid_y_test), max(valid_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()


