import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2
import rpy2.robjects as ro
from datetime import datetime


# R = 100 - optimal 
def calculate_kalman_points(game_data, R, P_0):
    # Calculate Q: variance of the differences between consecutive points
    Q = np.var(np.diff(game_data['PTS'])) / 3

    game_data = game_data.sort_values(by='GAME_DATE')

    # Kalman filter initialization
    n_timesteps = len(game_data) 
    x_hat = np.zeros(n_timesteps)  # a posteriori estimate of the state
    P = np.zeros(n_timesteps)  # a posteriori error estimate
    x_hat_minus = np.zeros(n_timesteps)  # a priori estimate of the state
    P_minus = np.zeros(n_timesteps)  # a priori error estimate
    K = np.zeros(n_timesteps)  # Kalman gain

    x_hat[0] = game_data['PTS'].iloc[0]
    P[0] = 1.0

    previous_timestamp = game_data['GAME_DATE'].iloc[0]
    adaptive_factor = 0.1

    # Kalman filter loop
    for t in range(1, n_timesteps):
        current_timestamp = game_data['GAME_DATE'].iloc[t]
        delta_t = (current_timestamp - previous_timestamp).days
        previous_timestamp = current_timestamp

        # Prediction step
        x_hat_minus[t] = x_hat[t-1]
        P_minus[t] = P[t-1] + Q 

        # Update step
        K[t] = P_minus[t] / (P_minus[t] + R)
        x_hat[t] = x_hat_minus[t] + K[t] * (game_data['PTS'].iloc[t] - x_hat_minus[t])
        P[t] = (1 - K[t]) * P_minus[t]

        residual = game_data['PTS'].iloc[t] - x_hat[t]
        R = (1 - adaptive_factor) * R + adaptive_factor * residual**2
        Q = (1 - adaptive_factor) * Q + adaptive_factor * (x_hat[t] - x_hat_minus[t])**2


    # Predict the next game points
    x_hat_next = x_hat[-1] # no th
    predicted_next_game_points = x_hat_next  # This is the predicted points for the next game

    return predicted_next_game_points, x_hat

def grid_search_kalman(game_data, R_values, P_0_values):
    best_R = None
    best_P_0 = None
    best_mae = float('inf')
    best_predictions = None

    for R in R_values:
        for P_0 in P_0_values:
            _, x_hat = calculate_kalman_points(game_data, R, P_0)
            mae = np.mean(np.abs(x_hat - game_data['PTS'].values))

            if mae < best_mae:
                best_mae = mae
                best_R = R
                best_P_0 = P_0
                best_predictions = x_hat

    return best_R, best_P_0, best_mae, best_predictions

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

robjects.r(r_code)
player_index = robjects.globalenv['thing']

# Define the Python function to call the R function
def get_player_stats(first_name, last_name):
    r_get_player_stats = robjects.globalenv['get_player_stats']
    stats = r_get_player_stats(player_index, first_name, last_name)
    return stats

def calculate_ed_points(game_data, current_date, final_delta_t):
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
    beta = 0.99

    # Convert date strings to datetime objects
    game_dates = game_data['GAME_DATE'].values

    # Calculate the number of days ago each game took place
    days_ago = [(current_date - game_date).astype('timedelta64[D]').astype(int) for game_date in game_dates]
    print(days_ago)
    points = game_data['PTS'].values
    print(points)

    # Calculate the weights and weighted points
    weights = [beta ** t for t in days_ago]
    return weighted_median(points, weights)

superstars = ['Napheesa Collier', 'Tina Charles', 'Caitlin Clark', 'Skylar Diggins-Smith', 'Ariel Atkins', 'Diana Taurasi', 'DiJonai Carrington']

for player in superstars:
    i = player.lstrip()
    first_name = player.split(" ")[0]
    last_name = player.split(" ")[1]

    player_stats = get_player_stats(first_name, last_name)
    gamelog = ro.conversion.rpy2py(player_stats)
    gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'], format="%b %d, %Y")
    # Why is it not
    gamelog['FGA'] = gamelog['FGA'].astype(float)
    gamelog['MIN'] = gamelog['MIN'].astype(float)
    gamelog['PTS'] = gamelog['PTS'].astype(float)
    
    print(gamelog)
    current_date = np.datetime64(datetime.now())
    points = calculate_ed_points(gamelog, current_date, 1)
    print(points)

    # Hyperparameter tuning on R and P_0
    """
    R_values = np.linspace(1, 1000, 20)  # Example values for R
    P_0_values = np.linspace(1, 5, 2)  # Example values for P_0

    # print(gamelog)

    print(np.var(gamelog["PTS"]))
    print(np.var(np.diff(gamelog["PTS"])))

    best_R, best_P_0, best_mae, best_predictions = grid_search_kalman(gamelog, (100, 150, 200), P_0_values)

    print(f"Best R for {player}: {best_R}")
    print(f"Best P_0 for {player}: {best_P_0}")
    print(f"Best MAE for {player}: {best_mae}")
    print(f"Predictions: {best_predictions}")
    print(len(best_predictions))

    """



