import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tqdm import tqdm
import warnings
from dataloading import load_regression_data, drop_regression_stats

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 64
np.random.seed(random_seed)

data = load_regression_data(league)
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

drop_regression_stats(data)
data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)
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
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=.2, shuffle=True)

    train = xgb.DMatrix(x_train, label=y_train, missing=-np.inf)
    test = xgb.DMatrix(x_test, label=y_test, missing=-np.inf)
    evals = [(train, 'train'), (test, 'test')]

    param = {
        'objective': 'reg:quantileerror',
        "quantile_alpha": quantiles,
        'max_depth': 3,
        'eta': 0.05,
        'subsample': 0.8
    }

    #  'interaction_constraints': [["L10 Median", "Minutes Diff"], ["L10 Median", "Rest Days"]]
    # can set feature_weights too

    model = xgb.train(param, train, num_boost_round=1000, early_stopping_rounds=40, evals=evals, verbose_eval=0)

    print("Best iteration: ", model.best_iteration)
    
    if model.best_iteration <= 30:
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

        # Compare the predictions to the lines to compute an accuracy when compared to over under
        y_lower = predictions[:, 0]  # alpha=0.476
        y_upper = predictions[:, 1]  # alpha=0.524

        # Log-transform the quantiles to fit a normal distribution
        log_y_lower = np.log(y_lower)
        log_y_upper = np.log(y_upper)

        # Calculate the mean (mu) and standard deviation (sigma) of the log-transformed data
        log_mu = (log_y_lower + log_y_upper) / 2
        log_sigma = (np.maximum(log_y_upper, log_y_lower) - np.minimum(log_y_upper, log_y_lower)) / (stats.norm.ppf(quantiles[1]) - stats.norm.ppf(quantiles[0]))

        # Check for valid sigma values
        assert np.all(log_sigma > 0), "Invalid sigma values: sigma must be positive"

        # Calculate the CDF for z_test using the log-normal distribution
        cdf_z_test = stats.lognorm.cdf(z_test, s=log_sigma, scale=np.exp(log_mu))

        # Print the results
        padding = 0.5
        valid_indices = np.where((z_test < np.minimum(y_upper, y_lower) - padding) | (z_test > np.maximum(y_lower, y_upper) + padding))[0]

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

        model.save_model(f'models/regression/{league}/XGBoost_{acc}%_OU.json')

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