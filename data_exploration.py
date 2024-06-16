import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


#data = pd.read_csv('data/wnba_train_over_under_data.csv')
data = pd.read_csv('data/wnba_train_regression.csv')
print(data.head())
OU = data['OU Result']
data.drop(['OU Result'], axis=1, inplace=True)
print(OU[:20])
print(data.describe())
class_counts = OU.value_counts()
print(class_counts)

print(mean_absolute_error(data['Points'], data['Line']))


"""

# Load the CSV file into a DataFrame
df = pd.read_csv('./data/wnba_correlations.csv')

# Basic info and statistics
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualizations
# Histograms
df.hist(bins=30, figsize=(10, 10))
plt.show()

# Boxplots
df.boxplot(column=['FGM T', 'FGA T', 'Minutes Diff', 'Rest Days', 'Recent T'], figsize=(10, 6))
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plots
sns.pairplot(df, x_vars=['FGM T', 'FGA T', 'Minutes Diff', 'Rest Days', 'Recent T'], y_vars='OU Result', height=5, aspect=0.8)
plt.show()

# Bar plot for the categorical variable 'Home'
sns.barplot(x='Home', y='OU Result', data=df)
plt.show()

# Feature Importance
# Correlation with target
print(df.corr()['OU Result'].sort_values(ascending=False))
"""