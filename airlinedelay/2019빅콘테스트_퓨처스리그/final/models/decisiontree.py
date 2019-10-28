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
reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr')

class decisiontree:
    def __init__(self, rfname):
        self.rfname = rfname
        dataframe = reader(rfname)

        #drop_columns = ['REG','IRR','ATT','STTATT']
        #df = dataframe.drop(drop_columns, axis=1)
        self.df = dataframe

        self.y = self.df['DLY']
        self.X = self.df.drop('DLY', axis=1)

    # oversamling
    def oversampling(self):
        self.over_x, self.over_y = ADASYN(random_state=0).fit_resample(self.X,self.y)

    # scaling
    def scaling(self):
        robust_scaler = RobustScaler()
        robust_scaler.fit(self.X)
        self.X = robust_scaler.transform(self.X)

    # split data to train and test
    def splitdata(self):
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.over_x, self.over_y, test_size=0.2, random_state=1, shuffle=True)
        self.dev_X, self.val_X, self.dev_y, self.val_y = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=1, shuffle=True)
        print('X_train : ',len(self.X_train))
        print('X_val : ',len(self.val_X))
        print('X_test :',len(self.X_test))

    #########################################################################
    # XGBoost
    def run_xgb(self, train_X, train_y, val_X, val_y, test_X):
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


    #########################################################################
    # LightGBM
    def run_lgb(self, train_X, train_y, val_X, val_y, test_X):
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

    ######################################################
    # CatBoost
    def run_cat(self, dev_X, dev_y, val_X, val_y, X_test):
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
        return pred_test_cat, cb_model


    ##############################################
    # Combine Predictions
    def combineprediction(self, pred_test_xgb, pred_test_lgb, pred_test_cat):
        sub = self.df[:]
        target = 'DLY'
        sub_xgb = pd.DataFrame()
        sub_xgb[target] = pred_test_xgb
        sub_lgb = pd.DataFrame()
        sub_lgb[target] = pred_test_lgb
        sub_cat = pd.DataFrame()
        sub_cat[target] = pred_test_cat
        sub[target] = (sub_lgb[target] * 0.25 + sub_xgb[target] * 0.25 + sub_cat[target] * 0.25)

        return sub, sub[target]

    def run(self):
        self.oversampling()
        self.scaling()
        self.splitdata()
        # Training XGB
        print("XGB Training Start...")
        pred_test_xgb, XGBoost_model = self.run_xgb(self.dev_X, self.dev_y, self.val_X, self.val_y, self.X_test)
        print("XGB Training Completed...")
        # Save XGB model
        joblib.dump(XGBoost_model, 'XGBoost_model.joblib.dat')

        # Traing LGB
        print("LightGBM Training Start...")
        pred_test_lgb, LightGBM_model, evals_result = self.run_lgb(self.dev_X, self.dev_y, self.val_X, self.val_y, self.X_test)
        print("LightGBM Training Completed...")
        # Save LightGBM Model
        joblib.dump(LightGBM_model, 'LightGBM_Model.pkl')

        # Training CB
        pred_test_cat, cb_model = self.run_cat(self, self.dev_X, self.dev_y, self.val_X, self.val_y, self.X_test)
        # Save CatBoost Model
        cb_model.save_model('CatBoost_Model')

        # Combine prediction
        sub, sub_target = self.combineprediction(pred_test_xgb, pred_test_lgb, pred_test_cat)
        print(sub_target)

if __name__ == "__main__":
    decisiontree('../merge_AFSNT_weather.csv').run()
