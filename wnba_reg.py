import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
import numpy as np
import random
from datetime import datetime
from dataloading import drop_unused_statistics
import rpy2
import rpy2.robjects as ro

def calculate_t_statistic(data, value, population_mean):
    # Calculate the sample standard deviation
    sample_std = np.std(data, ddof=1)  # ddof=1 to get the sample standard deviation

    # Calculate the sample size
    sample_size = len(data)

    # Compute the t-statistic manually
    t_statistic_manual = (value - population_mean) / (sample_std / np.sqrt(sample_size))
    return t_statistic_manual

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

#Calculate features given our odds
def calculate_wnba_features(df, today, home_teams, away_teams):
    superstars = df['Player']
    game_records_by_player = {}
    player_count = 0

    # Activate pandas2ri for seamless conversion between pandas and R data frames
    pandas2ri.activate()

    # Import the required R package
    wehoop = importr('wehoop')

    # Define the R function in Python
    r_code = """
    player_index <- wehoop::wnba_playerindex(season=wehoop::most_recent_wnba_season())
    thing <- player_index$PlayerIndex

    get_player_stats <- function(df, first_name, last_name) {
    # Find the player ID based on the first and last name
    player_ids <- which(df$PLAYER_LAST_NAME == last_name & df$PLAYER_FIRST_NAME == first_name)
    player_id <- df$PERSON_ID[player_ids]

    # Check if player ID is found
    if(length(player_id) == 0) {
        stop("Player not found in the dataframe")
    }

    # Print the player ID (optional)
    print(player_id)

    Sys.sleep(0.5)

    # Get the player's game log statistics for the most recent WNBA season
    stats <- wehoop::wnba_playergamelog(player_id = player_id, season = wehoop::most_recent_wnba_season())

    return(stats$PlayerGameLog)
    }
    """

    # Execute the R code to define the function and variables
    robjects.r(r_code)

    # Access the player index dataframe
    player_index = robjects.globalenv['thing']

    # Define the Python function to call the R function
    def get_player_stats(first_name, last_name):
        r_get_player_stats = robjects.globalenv['get_player_stats']
        stats = r_get_player_stats(player_index, first_name, last_name)
        return stats

    for i in superstars:
        #if player_count == 1:
            #break

        if i in game_records_by_player:
            continue
        
        i = i.lstrip()
        first_name = i.split(" ")[0]
        last_name = i.split(" ")[1]
        try:
            player_stats = get_player_stats(first_name, last_name)
            gamelog = ro.conversion.rpy2py(player_stats)
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'], format="%b %d, %Y")
            gamelog['FGA'] = gamelog['FGA'].astype(float)
            gamelog['MIN'] = gamelog['MIN'].astype(float)
            gamelog['PTS'] = gamelog['PTS'].astype(float)
            game_records_by_player[i] = gamelog
            gamelog.reset_index(drop=True, inplace=True)
            player_count += 1
        except Exception as e:
            print(e)
  
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
            sample_mean_fg_pct = gamelog.loc[row_index + 1 :, "FGA"].mean()
            # rolling_avg_fg_pct = gamelog.loc[row_index + 1 : row_index + 3, "FGA"].mean() 
            fg_games_list = gamelog.loc[row_index + 1 : row_index + 5, "FGA"].tolist()
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

    new_df['FG T'] = pd.to_numeric(new_df['FG T'], errors='coerce')
    new_df = new_df.fillna(0)
    new_df = new_df.astype(float)
    return new_df

if __name__ == "__main__":
    df = pd.read_csv('data/wnba_odds.csv')
    new_df = calculate_wnba_features(df, False, [], [])
    new_df.to_csv("data/wnba_train_regression.csv", index=False)