
"""
TODO: 


- SQLite to store all 2022-23 data 
- Load in and train XGBoost quantile regression


- Test correlation between EV and predictive accuracy
- Early stopping for NN training
- Combine create_dataset_from_odds and wnba_dataset

- Exploratory data analysis - PCA?

- New features
  - Opponent pace for WNBA (and NBA)
  - Opponent strength (apparently doesnt matter either)
  - Conditional on the game line LOL -- what was the ulimate point differential between the two teams. You can use the Vegas game line as a ground truth game line when making predictions
  
- New approach -- unified SARIMAX model using player-specific exogenous variables... 
   - Regressive based approach that is independent of the line might be more effective 

- Clean up and comment code
    - Pass in "DataLoader" object with more information so you don't have to pass in a bunch of props -- OOP principles
- include other odds for points

- Try fit transform on overall instead of calculting t-statistic

- Long term
    - SQL Database

"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from get_odds import get_odds_today
from nba_reg import calculate_features
from wnba_reg import calculate_wnba_features
from utils.odds import calculate_kelly_criterion
import os
import warnings

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

# from colorama import Fore, Style, init, deinit
# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
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

def calculate_ev(odds, probability):
    # Assume $100 bet
    # Calculate (Probability of Winning) x (Amount Won per Bet) – (Probability of Losing) x (Amount Lost per Bet)
    amount_won = 0
    if odds > 0:
        amount_won = odds
    else:
        amount_won = 100 * 100 / abs(odds)
    res = probability * amount_won - (1 - probability) * 100
    return res

# get odds
odds_today = get_odds_today(league)
if odds_today.empty:
    sys.exit(f"No {league} games for today")

print(odds_today)
data = calculate_features(odds_today, True, [], []) if league == "nba" else calculate_wnba_features(odds_today, True, [], [])
print(data)

# Get XG Boost model's predictions
model = xgb.Booster()
model.load_model(ou_model_name) 

predictions = model.predict(xgb.DMatrix(data, missing=-np.inf))

y_lower = predictions[:, 0]  # alpha=0.476
y_upper = predictions[:, 1]  # alpha=0.524

odds_today["Lower"] = y_lower
odds_today["Upper"] = y_upper

print(odds_today)


"""

y = []
ev = []
wagers = []

for i, z in enumerate(predictions):
    choice = i <= 
    kelly_fraction = calculate_kelly_criterion(odds_today.iloc[i]["Over"], z) if choice == 1 else calculate_kelly_criterion(odds_today.iloc[i]["Under"], 1 - z)
    expected_value = calculate_ev(odds_today.iloc[i]["Over"], z) if choice == 1 else calculate_ev(odds_today.iloc[i]["Under"], 1 - z)
    y.append(choice)
    ev.append(expected_value)
    wagers.append(kelly_fraction)

odds_today["Prediction"] = y
odds_today["EV"] = ev
odds_today["Wager Fraction"] = wagers

print(odds_today)
"""