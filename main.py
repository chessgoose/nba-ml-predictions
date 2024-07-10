
"""
TODO: 
- Calculate exponentional decay average of points
- Fix neural network not training issues... 
- Design frontend interface
- Revise scraping methodology so that it's more consistent (handle missing odds in row -- this is the biggest issue)

- Combine create_dataset_from_odds and wnba_dataset
- New features
    - Travel distance between
    - Opponent pace for WNBA (and NBA) (difference in my team vs opponent pace too)
    - Conditional on the game line LOL -- what was the ulimate point differential between the two teams. You can use the Vegas game line as a ground truth game line when making predictions

- Clean up and comment code
    - Pass in "DataLoader" object with more information so you don't have to pass in a bunch of props -- OOP principles
- Include other odds for points

- Try fit transform on overall instead of calculting t-statistic

- Long term - SQL Databases

- PCA ? 
- Implement pinball loss
- Look at 2022-23 NBA season stats to determine which factors are most useful for predicting points
- New approach -- unified SARIMAX model using player-specific exogenous variables... 
   - Regressive based approach that is independent of the line might be more effective 
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from get_odds import get_odds_today
from create_dataset_from_odds import calculate_features
from wnba_dataset import calculate_wnba_features
from nba_api.live.nba.endpoints import scoreboard
from utils.odds import calculate_kelly_criterion
import os
import warnings

league = "nba"
warnings.simplefilter(action='ignore', category=FutureWarning)

# from colorama import Fore, Style, init, deinit
# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
def get_best_model(league):
    new_file = ""   
    best_accuracy = 0.0
    for model_name in os.listdir(f'models/{league}'):
        model_accuracy = float(model_name.split("%")[0].split("_")[1])
        if model_accuracy > best_accuracy:
            new_file = model_name
            best_accuracy = model_accuracy
    return f'models/{league}/' + new_file

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

# Home teams -- not relevant rn so it doens't really work
home_teams = []
away_teams = []
try:
    games = scoreboard.ScoreBoard()
    d = games.get_dict()

    for game in d["scoreboard"]["games"]:
        home_teams.append(game["homeTeam"]["teamTricode"])
        away_teams.append(game["awayTeam"]["teamTricode"])
except:
    sys.exit("Failed to get home and away teams")

print("Home teams: ", home_teams)
print("Away teams; ", away_teams)
data = calculate_features(odds_today, True, home_teams, away_teams) if league == "nba" else calculate_wnba_features(odds_today, True, home_teams, away_teams)
print(data)

# Get XG Boost model's predictions
model = xgb.Booster()
model.load_model(ou_model_name) 

predictions = model.predict(xgb.DMatrix(data))
print(predictions) 

y = []
ev = []
wagers = []

for i, z in enumerate(predictions):
    choice = round(z)
    kelly_fraction = calculate_kelly_criterion(odds_today.iloc[i]["Over"], z) if choice == 1 else calculate_kelly_criterion(odds_today.iloc[i]["Under"], 1 - z)
    expected_value = calculate_ev(odds_today.iloc[i]["Over"], z) if choice == 1 else calculate_ev(odds_today.iloc[i]["Under"], 1 - z)
    y.append(choice)
    ev.append(expected_value)
    wagers.append(kelly_fraction)

odds_today["Prediction"] = y
odds_today["EV"] = ev
odds_today["Wager Fraction"] = wagers

print(odds_today)