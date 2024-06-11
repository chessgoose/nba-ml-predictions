import pandas as pd

data = pd.read_csv('data/wnba_train_regression.csv')
print(data.head())
OU = data['OU Result']
data.drop(['OU Result'], axis=1, inplace=True)
print(OU[:20])
print(data.describe())
class_counts = OU.value_counts()
print(class_counts)
