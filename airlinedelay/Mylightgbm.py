import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('testdata_3.csv')
#불필요한 컬럼
rm_col = ["REG","FLO"]
X_train = train_df.drop(rm_col+["DLY"], axis=1)
y_train = np.log1p(train_df["DLY"].values)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
X_test = test_df.drop(["ID"], axis=1)

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 40,
        "learning_rate": 0.004,
        "bagging_fraction": 0.6,
        "feature_fraction": 0.6,
        "bagging_frequency": 6,
        "bagging_seed": 42,
        "verbosity": -1,
        "seed": 42
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000,
                      valid_sets=[lgtrain, lgval],
                      early_stopping_rounds=100,
                      verbose_eval=150,
                      evals_result=evals_result)

    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result

pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)

gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(),
                   'split':model.feature_importance('split'),
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:50])



def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'eta': 0.001,
              'max_depth': 10,
              'subsample': 0.6,
              'colsample_bytree': 0.6,
              'alpha': 0.001,
              'random_state': 42,
              'silent': True}

    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds=100, verbose_eval=100)

    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

    return xgb_pred_y, model_xgb

pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)

