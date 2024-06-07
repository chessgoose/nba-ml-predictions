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
def drop_unused_statistics(df):
    # WNBA
    df.drop(["UR", "Home", "L5UR", "Line T", "Recent T"], axis=1, inplace=True)

    # NBA
    # df.drop(["UR", "Home", "L5UR", 'Rest Days', "Line T", "Recent T"], axis=1, inplace=True)

    # df.drop(["L5UR", "Home", "Recent T"], axis=1, inplace=True)
    # df.drop(['UR', 'Recent T'], axis=1, inplace=True)

# wnba max - 63.
def load_data(league="nba"):
    #scaler = MinMaxScaler()
    print(f"Loading data for {league}")
    data = pd.read_csv('data/train_playoffs_over_under_data.csv') if league == "nba" else pd.read_csv('data/wnba_train_over_under_data.csv') 

    OU = data['OU Result']
    # "FG T" and "OU Result"
    drop_unused_statistics(data)
    print(data.head())

    data.drop(["OU Result"], axis=1, inplace=True)
    # data['UR'] = scaler.fit_transform(data[['UR']])
    print(f'Number of rows: {len(data.index)}')

    data = data.astype(float)

    return data, OU

