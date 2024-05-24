import pandas as pd


# 56.5% -  ["L5UR", "UR", "Recent T", "OU Result"]
# drop home
# ["FG Diff", "Home", "Rest Days", "L5UR", "UR", "Recent T", "OU Result"]

def load_data():
    data = pd.read_csv('data/train_playoffs_over_under_data.csv')
    OU = data['OU Result']
    print(data.head())
    data.drop([ "L5UR", "OU Result"], axis=1, inplace=True)
    print(OU[:20])
    data = data.astype(float)
    return data, OU