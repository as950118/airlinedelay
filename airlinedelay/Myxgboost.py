#-*- coding:utf-8 -*-
import numpy as np
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot as plt
import csv
reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))

class Myxgb_fit:
    def __init__(self, rfname):
        #data = reader(rfname)
        #data = xgb.DMatrix('{0}?format=csv&label_column=0'.format(rfname))
        dataset = pd.read_csv(rfname, delimiter=',', encoding='euc-kr')

        feature_name = dataset.columns.values
        data = dataset.get_values()
        len_data = 10000
        print(feature_name)
        print(data)

        feature_x = feature_name[:15]
        feature_y = feature_name[15]
        print(feature_x)
        print(feature_y)

        merge_data = pd.DataFrame(data, columns=feature_name)
        print(merge_data)
        #x_data = np.array([[e for e in elem[:15]] for elem in data[1:len_data]])
        #y_data = np.array([elem[15] for elem in data[:len_data]])
        #x_data = np.array([list(map(int, [e for e in elem[:18]] )) for elem in data[:len_data]])
        #y_data = np.array([int(elem[18]) for elem in data[:len_data]])
        x_data = [[e for e in elem[:15]] for elem in data[:len_data]]
        y_data = [elem[15] for elem in data[:len_data]]

        x_train, y_train, = x_data[:len(x_data):2], y_data[:len(y_data):2]
        x_test, y_test = x_data[1:len(x_data):2], y_data[1:len(y_data):2]

        data_train = dataset.iloc[:len(data):2]
        data_test = dataset.iloc[0]+dataset.iloc[1:len(data):2]
        print(data_train)
        print(data_test)
        label = np.random.randint(2, size=5)
        train_matrix = xgb.DMatrix(data_train, feature_names=feature_name, label=[0])
        test_matrix = xgb.DMatrix(data_test, feature_names=feature_name, label = [0])
        params = {
            'colsample_bynode': 0.8,
            'learning_rate': 1,
            'max_depth': 5,
            'num_parallel_tree': 100,
            'objective': 'binary:logistic',
            'subsampling': 0.8,
            'tree_method': 'gpu_hist'
        }
        mod = xgb.train(dtrain=data_train, params=params)
        print(mod)
        model = xgb.XGBClassifier(feature_name=feature_name)
        print()
        model.fit(x_train, y_train)
        pre = model.predict(x_test)
        print(pre)

        xgb.plot_importance(model)
        plt.show()

if __name__ == "__main__":
    Myxgb_fit('testdata_3.csv')