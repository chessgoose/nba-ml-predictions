import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data
import warnings

"""
https://github.com/slavafive/ML-medium/blob/master/quantile_regression.ipynb
"""
league = "nba"
warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 1234
np.random.seed(random_seed)

data = load_regression_data()
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)
print(data.head())

acc_results = []

for x in tqdm(range(15)):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=.2, shuffle=True)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    evals = [(train, 'train'), (test, 'eval')]

    # grid search?
    param = {
        'objective': 'reg:quantileerror',
        "tree_method": "hist",
        "quantile_alpha": 0.5,
        'max_depth': 3,
        'eta': 0.075,
        'subsample': 1.0
    }

    model = xgb.train(param, train, \
                    num_boost_round=600, early_stopping_rounds=40, \
                    evals=evals, verbose_eval=0)
    
    print("Best iteration: ", model.best_iteration)

    if model.best_iteration <= 30:
        continue

    predictions = model.predict(test)

    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae}")

    # Compare the predictions to the lines to compute an accuracy when compared to over under
    predicted_ou_results = np.where(predictions > z_test, 1, 0)
    actual_ou_results = np.where(y_test > z_test, 1, 0)

    print(predicted_ou_results)
    print(actual_ou_results)

    acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
    print(f"Accuracy: {acc}%")
    acc_results.append(mae)

    # only save results if they are the best so far
    if mae == min(acc_results):
        feature_important = model.get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        scores = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        print(scores)

        model.save_model('models/regression/nba_XGBoost_{}%_OU.json'.format(acc))

