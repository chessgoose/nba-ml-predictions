import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# 56.5% -  ["L5UR", "UR", "Recent T", "OU Result"]
# drop home
# ["FG T", "Home", "Rest Days", "L5UR", "UR", "Recent T", "OU Result"]
def drop_unused_statistics(df):
    df.drop(['Line T', 'UR', 'FG T'], axis=1, inplace=True)

def load_data():
    #scaler = MinMaxScaler()
    data = pd.read_csv('data/train_playoffs_over_under_data.csv')
    OU = data['OU Result']
    print(data.head())
    # "FG T" and "OU Result"
    # 72.1% model: new_df.drop(['Line T', 'L5UR', 'FG T'], axis=1, inplace=True)
    drop_unused_statistics(data)
    data.drop(["OU Result"], axis=1, inplace=True)
    # data['UR'] = scaler.fit_transform(data[['UR']])
    print(OU[:20])
    data = data.astype(float)
    return data, OU

