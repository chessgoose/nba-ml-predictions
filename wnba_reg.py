import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
import numpy as np
import random
from datetime import datetime
from dataloading import drop_unused_statistics
from scipy.stats import ttest_ind
import rpy2
import rpy2.robjects as ro
import math

# This is likely how draftkings 
def calculate_ed_points(game_data, current_date):
    # Calculate ED points
    def weighted_median(data, weights):
        # Convert data and weights to numpy arrays
        data = np.array(data)
        weights = np.array(weights)
        
        # Sort data and weights based on data values
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Compute the cumulative sum of the weights
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find the index where the cumulative sum reaches or exceeds half the total weight
        half_weight = np.sum(weights) / 2.0
        median_index = np.where(cumulative_weights >= half_weight)[0][0]
        
        # Return the corresponding data value as the weighted median
        return sorted_data[median_index]
        
    # Play around this value
    beta = 0.98

    # Convert date strings to datetime objects
    game_dates = game_data['GAME_DATE'].values

    # Calculate the number of days ago each game took place
    days_ago = [(current_date - game_date).astype('timedelta64[D]').astype(int) for game_date in game_dates]
    points = game_data['PTS'].values

    # Calculate the weights and weighted points
    weights = [beta ** t for t in days_ago]
    weighted_sum = np.sum(points * weights)
    sum_of_weights = np.sum(weights)
    
    
    weighted_mean = weighted_sum / sum_of_weights
    return weighted_mean
    """
    return weighted_median(points, weights)
    """

def calculate_kalman_points(game_data, final_delta_t):
    # Calculate Q: variance of the differences between consecutive points
    differences = np.diff(game_data['PTS']) / 3

    Q = np.var(differences, ddof=1)
    R = 150 # MEASUREMENT VARIANCE (variance in LINE)

    game_data = game_data.sort_values(by='GAME_DATE')

    # Kalman filter initialization
    n_timesteps = len(game_data) 
    x_hat = np.zeros(n_timesteps)  # a posteriori estimate of the state
    P = np.zeros(n_timesteps)  # a posteriori error estimate
    x_hat_minus = np.zeros(n_timesteps)  # a priori estimate of the state
    P_minus = np.zeros(n_timesteps)  # a priori error estimate
    K = np.zeros(n_timesteps)  # Kalman gain

    x_hat[0] = game_data['PTS'].iloc[0]
    P[0] = 5.0 # Estimate in error cannot get better than maybe 5 points

    previous_timestamp = game_data['GAME_DATE'].iloc[0]

    # Kalman filter loop
    for t in range(1, n_timesteps):
        current_timestamp = game_data['GAME_DATE'].iloc[t]
        #delta_t = (current_timestamp - previous_timestamp).days
        dalta_t = 1
        previous_timestamp = current_timestamp

        # Prediction step
        x_hat_minus[t] = x_hat[t-1]
        P_minus[t] = P[t-1] + Q 
        # P_minus[t] = P[t-1] + Q * delta_t / 3 # Adjust Q by delta_t

        # Update step
        K[t] = P_minus[t] / (P_minus[t] + R)
        x_hat[t] = x_hat_minus[t] + K[t] * (game_data['PTS'].iloc[t] - x_hat_minus[t])
        P[t] = (1 - K[t]) * P_minus[t]

    # Predict the next game points
    x_hat_next = x_hat[-1] # no th
    predicted_next_game_points = x_hat_next  # This is the predicted points for the next game

    return predicted_next_game_points

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
def calculate_wnba_features(df, today, matchups, expected_points):
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

    # For every row in player_index, try the player and see if it is valid 
    
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
    box <- box %>%
  mutate(pace = field_goals_attempted + 0.44 * free_throws_attempted - offensive_rebounds + turnovers)
    # get total number of points of this game 

    calculate_avg_points_before_date <- function(team_data, team, date_threshold) {
    date_threshold <- as.Date(date_threshold)

    avg_points_all <- team_data %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    avg_points_after <- team_data %>%
        filter(team_abbreviation == team, game_date < date_threshold) %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    return(avg_points_after)
    }

    calculate_avg_points_on_date <- function(team_data, team, date_threshold) {
    date_threshold <- as.Date(date_threshold)

    avg_points_all <- team_data %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    avg_points_after <- team_data %>%
        filter(team_abbreviation == team, game_date == date_threshold) %>%
        summarise(avg_points = mean(team_score, na.rm = TRUE)) %>%
        pull(avg_points)

    return(avg_points_after)
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

    def calculate_avg_points_on_date(team, date_threshold):
        r_get_player_stats = robjects.globalenv['calculate_avg_points_on_date']
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
    headers = ["L10 Median", "Kalman", "DARKO", "Relative Performance", "Home", "Recent T", "Spread"]
    num_features = len(headers)
    
    if not today:
        headers.append("Points")
        headers.append("Line")
        headers.append("OU Result")

    for index, row in df.iterrows():
        try:
            gamelog = game_records_by_player[row["Player"].lstrip()]
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])

            row_index = None
            if not today:
                row_index = gamelog.index[gamelog["GAME_DATE"] == row["Date"]].tolist()[0]
            else:
                row_index = -1
            
            opponent_ppg = -np.nan
            relative_strength = -np.nan
            relative_performance = 0.0
            spread = 0.0
            if not today:
                team = gamelog.loc[row_index, "MATCHUP"].split(" ")[2]
                team = team if team not in convert_team_abbreviations else convert_team_abbreviations[team]
                my_team = gamelog.loc[row_index, "MATCHUP"].split(" ")[0]
                my_team = my_team if my_team not in convert_team_abbreviations else convert_team_abbreviations[my_team]
                game_date = gamelog.loc[row_index, "GAME_DATE"].strftime('%Y-%m-%d')
                spread = calculate_avg_points_on_date(my_team, game_date) - calculate_avg_points_on_date(team, game_date) 
                relative_performance = calculate_avg_points_on_date(my_team, game_date) - calculate_avg_points_before_date(my_team, game_date) 
            else:
                # Find my team in the list of matchups
                my_team = gamelog.loc[0, "MATCHUP"].split(" ")[0]
                print(my_team)
                team = ""
                expected_total = None
                #print(matchups)
                for team1, team2 in matchups:
                    if team1 == my_team:
                        expected_total = expected_points[team1]
                        team = team2 if team2 not in convert_team_abbreviations else convert_team_abbreviations[team2]
                    elif team2 == my_team:
                        expected_total = expected_points[team2]
                        team = team1 if team1 not in convert_team_abbreviations else convert_team_abbreviations[team1]
                print(team)
                my_team = my_team if my_team not in convert_team_abbreviations else convert_team_abbreviations[my_team]
                game_date = datetime.today().strftime('%Y-%m-%d')
                #print(game_date)
                opponent_ppg = calculate_avg_points_before_date(team, game_date)
                print(opponent_ppg)
                my_ppg = calculate_avg_points_before_date(my_team, game_date)
                print(my_ppg)
                print(expected_total)
                relative_performance = expected_total - my_ppg
                print(relative_performance)

            last_5_minutes = gamelog.loc[row_index + 1: row_index + 5, "MIN"].values
            print(last_5_minutes)
            #past_minutes = gamelog.loc[row_index + 3 : row_index + 10, "MIN"].values
            #print(past_minutes)
            
            #if len(past_minutes) < 5:
                #continue

            difference_mins = np.mean(last_5_minutes)
            #print(difference_mins)

            home = 1
            if not today:
                home = 0 if "@" in gamelog.loc[row_index, "MATCHUP"] else 1

            # Overall under rate and variance
            past_games = gamelog.loc[row_index + 1 :,  "PTS"].values
            last_10_median = np.median(past_games[:10])
            print(past_games)


            # We don't have enough data to extract anything meaningful
            # 8 is a reasonable number because 8 games should be enough to 
            if len(past_games) < 8:
                continue

            last_5_points = gamelog.loc[row_index + 1 : row_index + 5, "PTS"].values

            recent_t_statistic, p_points = ttest_ind(last_5_points, past_games, equal_var=False)

            # Calculate rest days since the last games
            rest_days = 0
            if not today:
                rest_days = (gamelog.loc[row_index, 'GAME_DATE'] - gamelog.loc[row_index + 1, 'GAME_DATE']).days
            else:
                rest_days = (pd.to_datetime("now") - gamelog.loc[0, 'GAME_DATE']).days
            print("Rest days: ", rest_days)

            # reversed_games = past_games[::-1]
            # print(gamelog.loc[row_index, 'GAME_DATE'])
            #print(gamelog.tail(len(past_games)))

            # needs to be properly sorted 
            last_games = gamelog.tail(len(past_games))
            kalman = calculate_kalman_points(last_games, final_delta_t=rest_days)
            # if not today else np.datetime64(datetime.now())
            ed = calculate_ed_points(last_games, current_date=np.datetime64(gamelog.loc[row_index, 'GAME_DATE'])) if not today else calculate_ed_points(last_games, current_date=np.datetime64(datetime.now()))

            if not today:
                OU_result = (gamelog.loc[row_index, 'PTS'] > row["Line"]).astype(int)
                dataset.append([last_10_median, kalman, ed, relative_performance, home, recent_t_statistic, spread, gamelog.loc[row_index, 'PTS'], row["Line"], OU_result])
            else:
                # ["L10 Median", "Kalman", "DARKO", "Relative Performance", "Rest Days", "Recent T", "Spread"]
                dataset.append([last_10_median, kalman, ed, relative_performance, home, recent_t_statistic, spread])
        except:
            if today:
                dataset.append([0 in range(num_features)])
            print("Failed finding record for player: " + row["Player"])
            # don't add a row
            #dataset.append([0, 0, 0, 0])

    new_df = pd.DataFrame(dataset, columns=headers)
    new_df = new_df.astype(float)
    if not today:
        drop_regression_stats(new_df)

    return new_df

if __name__ == "__main__":
    df = pd.read_csv('data/wnba_odds.csv')
    new_df = calculate_wnba_features(df, False, [], {})
    new_df.to_csv("data/wnba_train_regression.csv", index=False)