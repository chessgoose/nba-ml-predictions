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
    superstars = df['PERSON_ID']
    game_records_by_player = {}
    player_count = 0

    # Activate pandas2ri for seamless conversion between pandas and R data frames
    pandas2ri.activate()

    # Import the required R package
    wehoop = importr('wehoop')

    # Define the R code in Python
    r_code = """
    player_index <- wehoop::wnba_playerindex(season=(2023))
    thing <- player_index$PlayerIndex

    # TODO: take all players in this tibble who were active in 2023 and whose average points were ABOVE 7 ppg 
    get_player_stats <- function(player_id) {
    player_id <- strtoi(player_id)
    # Get the player's game log statistics for the most recent WNBA season
    stats <- wehoop::wnba_playergamelog(player_id = player_id, season =(2023))

    # Create a new row consisting of the following statistics
    return(stats$PlayerGameLog)
    }

    library(dplyr)

    box <- wehoop::load_wnba_team_box(seasons=(2023))

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
    def get_player_stats(player_id):
        r_get_player_stats = robjects.globalenv['get_player_stats']
        stats = r_get_player_stats(player_id)
        return stats

    for i in superstars:
        player_index = int(i)
        
        #if player_count == 2:
           #break

        if i in game_records_by_player:
            continue
        
        try:
            time.sleep(2)
            player_stats = get_player_stats(player_index)
            gamelog = ro.conversion.rpy2py(player_stats)
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'], format="%b %d, %Y")
            gamelog['FGA'] = gamelog['FGA'].astype(float)
            gamelog['MIN'] = gamelog['MIN'].astype(float)
            gamelog['PTS'] = gamelog['PTS'].astype(float)
            season_points = gamelog["PTS"].values
            season_avg = np.mean(season_points)
            if season_avg < 7.5:
                continue
            
            #print(gamelog)
            gamelog.reset_index(drop=True, inplace=True)
            game_records_by_player[i] = gamelog
            player_count += 1
    
        except Exception as e:
            print(f'Exception {e}')
  
    print(f"Collected records for {player_count} players")
    
    dataset = []
    headers = ["Date", "Player ID", "L10 Median", "Kalman", "DARKO", "Relative Performance", "Home", "Recent T", "Spread"]
    num_features = len(headers)
    
    if not today:
        headers.append("Points")

    for player_i in superstars:
        if player_i not in game_records_by_player:
            continue

        gamelog = game_records_by_player[player_i]
        season_points = gamelog["PTS"].values

        for row_index in range(len(season_points)):
            try:
                opponent_ppg = -np.nan
                relative_strength = -np.nan
                relative_performance = 0.0
                spread = 0.0

                past_games = gamelog.loc[row_index + 1 :,  "PTS"].values

                if len(past_games) < 8:
                    continue

                team = gamelog.loc[row_index, "MATCHUP"].split(" ")[2]
                team = team if team not in convert_team_abbreviations else convert_team_abbreviations[team]
                my_team = gamelog.loc[row_index, "MATCHUP"].split(" ")[0]
                my_team = my_team if my_team not in convert_team_abbreviations else convert_team_abbreviations[my_team]
                game_date = gamelog.loc[row_index, "GAME_DATE"].strftime('%Y-%m-%d')
                spread = calculate_avg_points_on_date(my_team, game_date) - calculate_avg_points_on_date(team, game_date) 
                relative_performance = calculate_avg_points_on_date(my_team, game_date) - calculate_avg_points_before_date(my_team, game_date) 

                home = 1
                if not today:
                    home = 0 if "@" in gamelog.loc[row_index, "MATCHUP"] else 1

                # Overall under rate and variance
                last_10_median = np.median(past_games[:10])
     
                last_5_points = gamelog.loc[row_index + 1 : row_index + 5, "PTS"].values

                recent_t_statistic, p_points = ttest_ind(last_5_points, past_games, equal_var=False)

                # needs to be properly sorted 
                last_games = gamelog.tail(len(past_games))
                rest_days = 0
                if not today:
                    rest_days = (gamelog.loc[row_index, 'GAME_DATE'] - gamelog.loc[row_index + 1, 'GAME_DATE']).days
                else:
                    rest_days = (pd.to_datetime("now") - gamelog.loc[0, 'GAME_DATE']).days

                # print(f'Rest Days: {rest_days}')

                kalman = calculate_kalman_points(last_games, final_delta_t=rest_days)
                # if not today else np.datetime64(datetime.now())
                ed = calculate_ed_points(last_games, current_date=np.datetime64(gamelog.loc[row_index, 'GAME_DATE'])) if not today else calculate_ed_points(last_games, current_date=np.datetime64(datetime.now()))

                if not today:
                    dataset.append([gamelog.loc[row_index, "GAME_DATE"].strftime('%Y-%m-%d'), player_i, last_10_median, kalman, ed, relative_performance, home, recent_t_statistic, spread, gamelog.loc[row_index, 'PTS']])

            except:
                if today:
                    dataset.append([0 in range(num_features)])
                print("Failed finding record for player: " + str(i))
                # don't add a row
                #dataset.append([0, 0, 0, 0])

    new_df = pd.DataFrame(dataset, columns=headers)
    return new_df

if __name__ == "__main__":
    df = pd.read_csv('data/person_ids.csv')
    new_df = calculate_wnba_features(df, False, [], {})
    new_df.to_csv("data/2023_wnba_season.csv", index=False)