import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import time

#players we want to look into
df = pd.read_csv('data/odds.csv')
superstars = df['Player']

#pull player ids from superstars
player_ids = []
for i in superstars:
    try:
        thing = players.find_players_by_full_name(i)[0]['id']
        player_ids.append(thing)
    except:
        player_ids.append(None)


#pull game logs of each player
game_logs =[]
for i in player_ids:
    print(i)
    if i is not None:
        time.sleep(0.5)
        game_logs.append(playergamelog.PlayerGameLog(player_id=i, season='2023').get_data_frames()[0])
    else:
        game_logs.append([])

player_data = dict(zip(superstars, game_logs))

data = []
headers = ["Player", "Game 1", "Game 2", "Game 3", "Game 4", "Game 5"]

for superstar in superstars:
    try:
        gamelog = player_data[superstar]
        points = gamelog['PTS'].tolist()[:5]
        row_data = [superstar] + points
        data.append(row_data)
    except:
        data.append([superstar, 0, 0, 0, 0, 0])

new_df = pd.DataFrame(data, columns=headers)
new_df.to_csv("data/points.csv")
