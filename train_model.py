

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_data
import warnings

league = "nba"

warnings.simplefilter(action='ignore', category=FutureWarning)

random_seed = 1234
np.random.seed(random_seed)

data, OU = load_data(league)
acc_results = []

for x in tqdm(range(50)):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.2, shuffle=True)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    evals = [(train, 'train'), (test, 'eval')]

    param = {
        'max_depth': 3,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'subsample': 0.8
    }

    model = xgb.train(param, train, \
                    num_boost_round=500, early_stopping_rounds=50, \
                    evals=evals, verbose_eval=0)
    
    print("Best iteration: ", model.best_iteration)

    if model.best_iteration <= 30:
        continue

    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(round(z))

    acc = round(accuracy_score(y_test, y) * 100, 1)
    print(f"Accuracy: {acc}%")
    acc_results.append(acc)

    # Calculate f1 score 
    f1 = f1_score(y_test, y, average='weighted')  # Use 'weighted' to handle class imbalance
    print(f"F1 Score: {f1:.2f}")
    
    # only save results if they are the best so far
    if acc == max(acc_results):
        feature_important = model.get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        scores = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        print(scores)

        model.save_model('models/' + league + '/XGBoost_{}%_OU.json'.format(acc))

