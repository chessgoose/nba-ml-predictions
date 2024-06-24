import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from get_odds import get_odds_today, get_matchups
from dataloading import drop_regression_stats
from nba_reg import calculate_features
from wnba_reg import calculate_wnba_features
from utils.odds import calculate_kelly_criterion
from train_nn_regressor import QuantileRegressor
import os
import warnings
from colorama import Fore, Style, init, deinit

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_best_model(league):
    new_file = ""   
    best_accuracy = 0.0
    for model_name in os.listdir(f'models/regression/{league}/'):
        model_accuracy = float(model_name.split("%")[0].split("_")[1])
        if model_accuracy > best_accuracy:
            new_file = model_name
            best_accuracy = model_accuracy
    return f'models/regression/{league}/' + new_file

ou_model_name = get_best_model(league)
print(ou_model_name)

# get odds
odds_today = get_odds_today(league)
if odds_today.empty:
    sys.exit(f"No {league} games for today")

print(odds_today)

matchups = get_matchups()
print(matchups)
data = calculate_features(odds_today, True, [], []) if league == "nba" else calculate_wnba_features(odds_today, True, matchups)
print(data)
drop_regression_stats(data)

# Get XG Boost model's predictions
model = xgb.Booster()
model.load_model(ou_model_name) 

predictions = model.predict(xgb.DMatrix(data, missing=-np.inf))

y_lower = predictions[:, 0]  # alpha=0.476
y_upper = predictions[:, 1]  # alpha=0.524

odds_today["Lower XGBoost"] = y_lower
odds_today["Upper XGBoost"] = y_upper

# Initialize colorama
init(autoreset=True)

def get_row_color(lower, upper, total):
    if min(lower, upper) > total + 0.5:
        return Fore.GREEN
    elif max(lower, upper) < total - 0.5:
        return Fore.RED
    else:
        return ''

# Print each row with appropriate highlighting
for index, row in odds_today.iterrows():
    lower = row["Lower XGBoost"]
    upper = row["Upper XGBoost"]
    total = row["Line"]
    color = get_row_color(lower, upper, total)

    print(color + f"{row['Date']} - {row['Player']} - Line: {total} - Over: {row['Over']} - Under: {row['Under']} - Lower XGBoost: {lower:.2f} - Upper XGBoost: {upper:.2f}" + Style.RESET_ALL)

# Deinitialize colorama
deinit()