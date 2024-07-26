import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# 50.7% with all statistics
# 57.0%, 62.9% - df.drop(['UR', 'Recent T'], axis=1, inplace=True)

# 54% with minutes diff 
# 53% w FG T
# 50% with Home 
# 50% with Rest Days
# 51.7% with L5UR
# 54.4% with UR
# 48% with Recent T
# 49.7% with Line T

# ["FG T", "Home", "Rest Days", "L5UR", "UR", "Recent T", "OU Result"]
#df.drop(['Line T', 'UR'], axis=1, inplace=True)
def drop_unused_statistics(df, league):
    # WNBA

    # percentage of unders?

    if league == "wnba":
        df.drop(["FG T", "eFG T", "Minutes Diff", "Rest Days", "L5UR", "UR", "Recent T", "Line T"], axis=1, inplace=True)
    else:
        df.drop(["L5UR", "UR", "Home", "L5UR", 'Rest Days', "Recent T", "Line T"], axis=1, inplace=True)

    # df.drop(["L5UR", "Home", "Recent T"], axis=1, inplace=True)
    # df.drop(['UR', 'Recent T'], axis=1, inplace=True)

# wnba max - 63.
def load_data(league="nba"):
    #scaler = MinMaxScaler()
    print(f"Loading data for {league}")
    data = pd.read_csv('data/train_playoffs_over_under_data.csv') if league == "nba" else pd.read_csv('data/wnba_train_over_under_data.csv') 
    #data = pd.read_csv('data/wnba_train_regression.csv') 
    OU = data['OU Result']
    # OU = data['Points']

    
    # "FG T" and "OU Result"
    drop_unused_statistics(data, league)
    print(data.head())

    data.drop(["OU Result"], axis=1, inplace=True)
    # data.drop(["OU Result", "Points", "Line"], axis=1, inplace=True)
    # data['UR'] = scaler.fit_transform(data[['UR']])
    print(f'Number of rows: {len(data.index)}')

    data = data.astype(float)

    return data, OU

def load_regression_data(league):
    data = pd.read_csv(f'data/{league}_train_regression.csv')
    points = data["Points"]

    # data['UR'] = scaler.fit_transform(data[['UR']])
    print(f'Number of rows: {len(data.index)}')

    # data.drop(["Points", "Line", "OU Result"], axis=1, inplace=True)
    data = data.astype(float)

    return data

def drop_regression_stats(data):
    data.drop(["L10 Median", "Kalman", "Home", "Recent T", "Spread"], axis=1, inplace=True)

    #data.drop(["L10 Median", "Kalman", "Home", "Recent T", "Spread"], axis=1, inplace=True)

def load_2023_data():
    data = pd.read_csv('data/2023_wnba_season.csv')
    data.drop(['Player ID', 'Date'], axis=1, inplace=True)
    data = data.astype(float)

    return data

