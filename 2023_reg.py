import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tqdm import tqdm
import warnings
from dataloading import load_regression_data, drop_regression_stats, load_2023_data

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 42
np.random.seed(random_seed)

data = load_2023_data()
points = data["Points"]
print(data.head())

drop_regression_stats(data)
data.drop(["Points"], axis=1, inplace=True)
print(data.head())

quantiles = np.array([0.476, 0.524])

def calculate_coverage(predictions, y_test, quantiles):
    coverages = []
    for q, y_pred in zip(quantiles, predictions.T):
        coverage = np.mean(y_test <= y_pred)
        coverages.append(coverage)
    return coverages

def calculate_calibration_error(coverages, quantiles):
    return np.mean(np.abs(coverages - quantiles))

best_model = None
best_calibration_error = float('inf')
best_coverage_results = None

for x in tqdm(range(20)):
    # No shuffle -- inot shuffling causes it to not train
    x_train, x_test, y_train, y_test = train_test_split(data, points, test_size=.2)

    train = xgb.DMatrix(x_train, label=y_train, missing=-np.inf)
    test = xgb.DMatrix(x_test, label=y_test, missing=-np.inf)
    evals = [(train, 'train'), (test, 'test')]

    param = {
        'objective': 'reg:quantileerror',
        "quantile_alpha": quantiles,
        'max_depth': 3,
        'eta': 0.05,
        'subsample': 0.7
    }

    #  'interaction_constraints': [["L10 Median", "Minutes Diff"], ["L10 Median", "Rest Days"]]
    # can set feature_weights too

    model = xgb.train(param, train, num_boost_round=1000, early_stopping_rounds=50, evals=evals, verbose_eval=0)

    print("Best iteration: ", model.best_iteration)
    
    if model.best_iteration <= 50:
        continue

    predictions = model.predict(test)

    coverages = calculate_coverage(predictions, y_test, quantiles)
    calibration_error = calculate_calibration_error(coverages, quantiles)

    if calibration_error < best_calibration_error:
        feature_important = model.get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        scores = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        print(scores)

        best_calibration_error = calibration_error
        best_model = model
        best_coverage_results = coverages

        print(f"Best coverage error: {best_calibration_error}")

        model.save_model(f'models/regression/{league}/XGBoost_2023.json')

print(f"Best calibration error: {best_calibration_error}")
print(f"Best coverage results: {best_coverage_results}")

# Plot calibration curve for the best model
plt.figure(figsize=(8, 6))
plt.plot(quantiles, best_coverage_results, marker='o', label='Empirical coverage')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect calibration')
plt.xlabel('Quantiles')
plt.ylabel('Empirical Coverage Probability')
plt.title('Calibration Plot for Best Model')
plt.legend()
plt.show()