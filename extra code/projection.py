import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
import numpy as np

#players we want to look into
df = pd.read_csv('data/new_odds_two.csv')
superstars = df['Player']

#pull player ids from superstars
player_ids = []
for i in superstars:
    try:
        if len(player_ids) > 5:
            break
        if i == "Nicolas Claxton":
            i = "Nic Claxton"
        thing = players.find_players_by_full_name(i)[0]['id']
        if thing not in player_ids:
            player_ids.append(thing)
    except:
        player_ids.append(None)
   
# Make bet if our projection and bettingpros projection is the same

#pull game logs of each player
game_logs =[]
for i in player_ids:
    print(i)
    if i is not None:
        time.sleep(0.5)
        game_logs.append(playergamelog.PlayerGameLog(player_id=i, season='2023', season_type_all_star="Playoffs").get_data_frames()[0])
    else:
        game_logs.append([])

player_data = dict(zip(superstars, game_logs))

data = []
headers = ["Player", "Projection", "Standard Deviation"]

for superstar in superstars[0]:
    try:
        gamelog = player_data[superstar]
        points = gamelog['PTS'].tolist()[::-1]
        print(points)

        #Fit SES model
        model = SimpleExpSmoothing(points)
        fit_model = model.fit()

        # Predict the next element
        next_prediction = fit_model.predict(start=len(points), end=len(points))[0]
        
        std_deviation = np.std(points)

        row_data = [superstar, next_prediction, std_deviation]
        data.append(row_data)
    except:
        data.append([superstar, 0, 0])

df = pd.DataFrame(data, columns=headers)
df.to_csv("data/points_new.csv")
