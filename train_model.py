

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_data
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

data, OU = load_data()
acc_results = []

for x in tqdm(range(20)):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.15)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    evals = [(train, 'train'), (test, 'eval')]

    param = {
        'max_depth': 5,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'subsample': 0.8
    }

    model = xgb.train(param, train, \
                    num_boost_round=500, early_stopping_rounds=50, \
                    evals=evals, verbose_eval=0)
    
    print("Best iteration: ", model.best_iteration)

    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(round(z))

    acc = round(accuracy_score(y_test, y) * 100, 1)
    print(f"{acc}%")
    acc_results.append(acc)
    # only save results if they are the best so far
    if acc == max(acc_results):
        model.save_model('models/XGBoost_{}%_OU.json'.format(acc))

