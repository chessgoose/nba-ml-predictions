from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier  # or XGBRegressor for regression tasks
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloading import load_data

data, OU = load_data()
x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.2)

# Define the parameter grid
param_grid = {
    'eta': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'scale_pos_weight': [1, 2, 3]  # Adjust if dealing with imbalanced data
}

# Initialize the model
xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, 
                                   n_iter=100, scoring='accuracy', cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(x_train, y_train)

# Best parameters
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

best_model = random_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")