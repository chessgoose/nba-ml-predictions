import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data
import warnings
import matplotlib.pyplot as plt
import numpy as np

"""

https://github.com/slavafive/ML-medium/blob/master/quantile_regression.ipynb

"""

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 1234
np.random.seed(random_seed)

data = load_regression_data(league)
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)
print(data.head())

quantiles = np.array([0.476, 0.524])

acc_results = []
for x in tqdm(range(15)):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=.2, shuffle=True)

    train = xgb.DMatrix(x_train, label=y_train, missing=-np.inf)
    test = xgb.DMatrix(x_test, label=y_test, missing=-np.inf)
    evals = [(train, 'train'), (test, 'eval')]

    # grid search?
    param = {
        'objective': 'reg:quantileerror',
        "quantile_alpha": quantiles,
        'max_depth': 3,
        'eta': 0.05,
        'subsample': 1.0
    }

    model = xgb.train(param, train, \
                    num_boost_round=1000, early_stopping_rounds=40, \
                    evals=evals, verbose_eval=0)
    
    print("Best iteration: ", model.best_iteration)

    if model.best_iteration <= 30:
        continue

    predictions = model.predict(test)

    """
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae}")
    """

    # Compare the predictions to the lines to compute an accuracy when compared to over under
    y_lower = predictions[:, 0]  # alpha=0.476
    y_upper = predictions[:, 1]  # alpha=0.524

    valid_indices = np.where((z_test < y_lower) | (z_test > y_upper))[0]

    if len(valid_indices) == 0:
        print("No valid predictions outside the range [predictions_low, predictions_high]")
        continue

    valid_predictions = (y_lower[valid_indices] + y_upper[valid_indices]) / 2
    valid_y_test = y_test.iloc[valid_indices]
    valid_z_test = z_test.iloc[valid_indices]

    mae = mean_absolute_error(valid_y_test, valid_predictions)
    print(f"MAE: {mae}")

    # Compare the predictions to the lines to compute accuracy when compared to over/under
    predicted_ou_results = np.where(valid_predictions > valid_z_test, 1, 0)
    actual_ou_results = np.where(valid_y_test > valid_z_test, 1, 0)
    acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
    print(f"Accuracy: {acc}% on {len(predicted_ou_results)} results")
    acc_results.append(acc)

    # only save results if they are the best so far
    if acc == max(acc_results):
        feature_important = model.get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        scores = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        print(scores)

        plt.figure(figsize=(10, 6))
        plt.scatter(valid_y_test, valid_predictions, color='blue', label='Predictions')

        # Plot a line of perfect prediction for reference
        min_val = min(min(valid_y_test), min(valid_predictions))
        max_val = max(max(valid_y_test), max(valid_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

        # Add labels and title
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()

        # Show the plot
        plt.show()

        model.save_model(f'models/regression/{league}/XGBoost_{acc}%_OU.json')

