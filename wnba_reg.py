import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
import numpy as np
import random
from datetime import datetime
from dataloading import drop_unused_statistics
import rpy2
import rpy2.robjects as ro
import math

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

convert_team_abbreviations = {
    'CON': 'CONN',
    'NYL': 'NY',
    'LAS': 'LA',
    'LVA': 'LV',
    'PHO': 'PHX',
    'WAS': 'WSH'
}

#Calculate features given our odds
def calculate_wnba_features(df, today, matchups):
    superstars = df['Player']
    game_records_by_player = {}
    player_count = 0

    # Activate pandas2ri for seamless conversion between pandas and R data frames
    pandas2ri.activate()

    # Import the required R package
    wehoop = importr('wehoop')

    # Define the R code in Python
    r_code = """
    player_index <- wehoop::wnba_playerindex(season=wehoop::most_recent_wnba_season())
    thing <- player_index$PlayerIndex

    get_player_stats <- function(df, first_name, last_name) {
    # Find the player ID based on the first and last name
    player_ids <- which(df$PLAYER_LAST_NAME == last_name & df$PLAYER_FIRST_NAME == first_name)
    player_id <- df$PERSON_ID[player_ids]

    # Check if player ID is found
    if(length(player_id) == 0) {
        stop(paste("Player", first_name, last_name, "not found in the dataframe"))
    }

    # Print the player ID (optional)
    print(player_id)

    Sys.sleep(0.5)

    # Get the player's game log statistics for the most recent WNBA season
    stats <- wehoop::wnba_playergamelog(player_id = player_id, season = wehoop::most_recent_wnba_season())

    return(stats$PlayerGameLog)
    }

    library(dplyr)

    box <- wehoop::load_wnba_team_box()

    calculate_avg_points_before_date <- function(team_data, team, date_threshold) {
    date_threshold <- as.Date(date_threshold)

    avg_points_all <- team_data %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    avg_points_after <- team_data %>%
        filter(team_abbreviation == team, game_date < date_threshold) %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    return (avg_points_after - avg_points_all)
    }
    """

    # Execute the R code to define the function and variables
    robjects.r(r_code)

    # Access the player index dataframe
    player_index = robjects.globalenv['thing']
    team_data = robjects.globalenv['box']

    # Define the Python function to call the R function
    def calculate_avg_points_before_date(team, date_threshold):
        r_get_player_stats = robjects.globalenv['calculate_avg_points_before_date']
        stats = r_get_player_stats(team_data, team, date_threshold)
        return stats[0]

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
    headers = ["L10 Median", "Relative Strength", "Minutes Diff", "Rest Days", "Recent T", "Opponent PPG"]
    num_features = len(headers)
    
    if not today:
        headers.append("Points")
        headers.append("Line")
        headers.append("OU Result")

    for index, row in df.iterrows():
        try:
            gamelog = game_records_by_player[row["Player"].lstrip()]
            # print(gamelog)

            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])

            row_index = None
            if not today:
                row_index = gamelog.index[gamelog["GAME_DATE"] == row["Date"]].tolist()[0]
            else:
                row_index = -1
            
            opponent_ppg = -np.nan
            relative_strength = -np.nan
            if not today:
                team = gamelog.loc[row_index, "MATCHUP"].split(" ")[2]
                team = team if team not in convert_team_abbreviations else convert_team_abbreviations[team]
                my_team = gamelog.loc[row_index, "MATCHUP"].split(" ")[0]
                my_team = my_team if my_team not in convert_team_abbreviations else convert_team_abbreviations[my_team]
                game_date = gamelog.loc[row_index, "GAME_DATE"].strftime('%Y-%m-%d')
                opponent_ppg = calculate_avg_points_before_date(team, game_date)
                my_ppg = calculate_avg_points_before_date(my_team, game_date) 
                relative_strength = my_ppg - opponent_ppg
            else:
                # Find my team in the list of matchups
                my_team = gamelog.loc[0, "MATCHUP"].split(" ")[0]
                print(my_team)
                team = ""
                #print(matchups)
                for team1, team2 in matchups:
                    if team1 == my_team:
                        team = team2 if team2 not in convert_team_abbreviations else convert_team_abbreviations[team2]
                    elif team2 == my_team:
                        team = team1 if team1 not in convert_team_abbreviations else convert_team_abbreviations[team1]
                print(team)
                my_team = my_team if my_team not in convert_team_abbreviations else convert_team_abbreviations[my_team]
                game_date = datetime.today().strftime('%Y-%m-%d')
                #print(game_date)
                opponent_ppg = calculate_avg_points_before_date(team, game_date)
                print(opponent_ppg)
                my_ppg = calculate_avg_points_before_date(my_team, game_date) 
                print(my_ppg)
                relative_strength = my_ppg - opponent_ppg
                print(relative_strength)

            last_5_minutes = gamelog.loc[row_index + 1 : row_index + 5, "MIN"]
            past_minutes = gamelog.loc[row_index + 1 :, "MIN"]
            difference_mins = calculate_t_statistic(last_5_minutes, np.mean(last_5_minutes), past_minutes.mean())

            # Overall under rate and variance
            past_games = gamelog.loc[row_index + 1 :, "PTS"][:10]
            last_10_median = np.median(past_games)

            last_5_points = gamelog.loc[row_index + 1 : row_index + 5, "PTS"]
            games_list = last_5_points.tolist()
            recent_t_statistic = calculate_t_statistic(games_list, np.mean(games_list), past_games.mean())
            
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
                dataset.append([last_10_median, relative_strength, difference_mins, rest_days, recent_t_statistic, opponent_ppg, gamelog.loc[row_index, 'PTS'], row["Line"], OU_result])
            else:
                dataset.append([last_10_median, relative_strength, difference_mins, rest_days, recent_t_statistic, opponent_ppg])
        except:
            if today:
                dataset.append([0 in range(num_features)])
            print("Failed finding record for player: " + row["Player"])
            # don't add a row
            #dataset.append([0, 0, 0, 0])

    new_df = pd.DataFrame(dataset, columns=headers)

    # new_df['FG T'] = pd.to_numeric(new_df['FG T'], errors='coerce')
    # new_df = new_df.fillna(0)
    new_df = new_df.astype(float)
    return new_df

if __name__ == "__main__":
    df = pd.read_csv('data/wnba_odds.csv')
    new_df = calculate_wnba_features(df, False, [])
    new_df.to_csv("data/wnba_train_regression.csv", index=False)