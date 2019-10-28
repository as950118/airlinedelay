import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import RobustScaler
import joblib
import catboost
from catboost import CatBoostRegressor
from keras.models import load_model
import lstm_oversampling

reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr')

rfname = 'merge_AFSNT_weather.csv'
dataframe = reader(rfname)

#drop_columns = ['REG','IRR','ATT','STTATT']
#df = dataframe.drop(drop_columns, axis=1)
df = dataframe

y = df['DLY']
X = df.drop('DLY', axis=1)

# oversamling
over_x, over_y = ADASYN(random_state=0).fit_resample(X,y)

# scaling
robust_scaler = RobustScaler()
robust_scaler.fit(X)
X = robust_scaler.transform(X)

# split data to train and test
X_train, X_test, y_train, y_test= train_test_split(over_x, over_y, test_size=0.2, random_state=1, shuffle=True)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True)
print('X_train : ',len(X_train))
print('X_val : ',len(val_X))
print('X_test :',len(X_test))

#########################################################################
# XGBoost
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

# Training XGB
print("XGB Training Start...")
pred_test_xgb, XGBoost_model = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")

# Save XGB model
joblib.dump(XGBoost_model, 'XGBoost_model.joblib.dat')

#########################################################################
# LightGBM
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
print("LightGBM Training Start...")
pred_test_lgb, LightGBM_model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")

# Save LightGBM Model
joblib.dump(LightGBM_model, 'LightGBM_Model.pkl')

######################################################
# CatBoost
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
cb_model.fit(dev_X, dev_y,
             eval_set=(val_X, val_y),
             use_best_model=True,
             verbose=50)
pred_test_cat = np.expm1(cb_model.predict(X_test))
cb_model.save_model('CatBoost_Model')

##############################################
# Combine Predictions

sub = df[:]
target = 'DLY'
sub_xgb = pd.DataFrame()
sub_xgb[target] = pred_test_xgb
sub_lgb = pd.DataFrame()
sub_lgb[target] = pred_test_lgb
sub_cat = pd.DataFrame()
sub_cat[target] = pred_test_cat
sub[target] = (sub_lgb[target] * 0.25 + sub_xgb[target] * 0.25 + sub_cat[target] * 0.25)

CNNLSTMRadam_model = load_model('./models/LSTM_CNN_RAdam.h5')
print(CNNLSTMRadam_model['val_acc'])
