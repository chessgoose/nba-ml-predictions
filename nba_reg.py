import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
import numpy as np
import random
from datetime import datetime
from dataloading import drop_unused_statistics

def calculate_t_statistic(data, value, population_mean):
    # Calculate the sample standard deviation
    sample_std = np.std(data, ddof=1)  # ddof=1 to get the sample standard deviation

    # Calculate the sample size
    sample_size = len(data)

    # Compute the t-statistic manually
    t_statistic_manual = (value - population_mean) / (sample_std / np.sqrt(sample_size))
    return t_statistic_manual

#Calculate features given our odds
def calculate_features(df, today, home_teams, away_teams):
    superstars = df['Player']
    game_records_by_player = {}
    player_count = 0

    for i in superstars:
        #if player_count == 1:
            #break

        try:
            if i == "Nicolas Claxton":
                i = "Nic Claxton"

            if i in game_records_by_player:
                continue

            player_id = players.find_players_by_full_name(i)[0]['id']
            time.sleep(random.randint(1, 3))
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2023', season_type_all_star="Playoffs").get_data_frames()[0]
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'], format="%b %d, %Y")
            column_list = gamelog.columns.tolist()
            print(column_list)
            game_records_by_player[i] = gamelog
            player_count += 1
        except:
            continue
    
    print(f"Collected records for {player_count} players")

    dataset = []
    headers = ["L10 Median", "FG T", "Minutes Diff", "Rest Days"]
    num_features = len(headers)
    
    if not today:
        headers.append("Points")
        headers.append("Line")
        headers.append("OU Result")

    for index, row in df.iterrows():
        try:
            gamelog = game_records_by_player[row["Player"]]
            # print(gamelog)

            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])

            row_index = None
            if not today:
                row_index = gamelog.index[gamelog["GAME_DATE"] == row["Date"]].tolist()[0]
            else:
                row_index = -1

            # print("Row index:", row_index)
            # FG_pct

            # maybe calculate true shooting percentage instead... 
            sample_mean_fg_pct = gamelog.loc[row_index + 1 :, "FGM"].mean()
            # rolling_avg_fg_pct = gamelog.loc[row_index + 1 : row_index + 3, "FGA"].mean() 
            fg_games_list = gamelog.loc[row_index + 1 : row_index + 5, "FGM"].tolist()
            difference_fg = calculate_t_statistic(fg_games_list, np.mean(fg_games_list), sample_mean_fg_pct)
            #difference_fg = (rolling_avg_fg_pct - sample_mean_fg_pct) 

            #print("Average FG PCT: ", sample_mean_fg_pct)

            # TODO: calculate number of miles needed to travel from point a to point b if the previous game was further

            # TODO: calculate team pace 
            # https://www.nba.com/stats/teams/advanced?dir=-1&sort=PACE&SeasonType=Regular+Season

            # weekly_mean_points = gamelog["PTS"].rolling(3).mean()
            #sample_mean_mins = gamelog.loc[row_index + 1 :, "MIN"].mean()
            # how many minutes -- maybe make 4 games to reduce variance
            # Rolling average of the past 2 games minutes -- has this player been playing more minutes recently? 
            # rolling_avg_mins = gamelog.loc[row_index + 1 : row_index + 4, "MIN"].mean()  
            # difference_mins = rolling_avg_mins - sample_mean_mins
            last_5_minutes = gamelog.loc[row_index + 1 : row_index + 5, "MIN"]
            past_minutes = gamelog.loc[row_index + 1 :, "MIN"]
            difference_mins = calculate_t_statistic(last_5_minutes, np.mean(last_5_minutes), past_minutes.mean())

            # Overall under rate and variance
            past_games = gamelog.loc[row_index + 1 :, "PTS"][:10]
            last_10_median = np.median(past_games)
            
            # Calculate rest days since the last games
            rest_days = 0
            if not today:
                rest_days = (gamelog.loc[row_index, 'GAME_DATE'] - gamelog.loc[row_index + 1, 'GAME_DATE']).days
            else:
                rest_days = (pd.to_datetime("now") - gamelog.loc[0, 'GAME_DATE']).days
            print("Rest days: ", rest_days)
            
            # Skip that shit if for some reason we don't have valid shit (b/c then we can't really learn anything)
            if not today and difference_fg == 0 and difference_mins == 0:
                continue

            if not today:
                #headers = ["L10 Median", "FG T", "Minutes Diff", "Rest Days", "Points", "Line", "OU Result"]
                OU_result = (gamelog.loc[row_index, 'PTS'] > row["Line"]).astype(int)
                # headers = ["FG PCT", "Home", "Minutes Diff", "Rest Days", "L5UR", "UR"] 
                dataset.append([last_10_median, difference_fg, difference_mins, rest_days, gamelog.loc[row_index, 'PTS'], row["Line"], OU_result])
            else:
                dataset.append([last_10_median, difference_fg, difference_mins, rest_days])
        except:
            if today:
                dataset.append([0 in range(num_features)])
            print("Failed finding record for player: " + row["Player"])
            # don't add a row
            #dataset.append([0, 0, 0, 0])

    new_df = pd.DataFrame(dataset, columns=headers)
    return new_df

if __name__ == "__main__":
    df = pd.read_csv('data/new_odds_two.csv')
    new_df = calculate_features(df, False, [], [])
    new_df.to_csv("data/nba_train_regression.csv", index=False)