import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import csv
reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))
'''
rfname = 'testdata_2.csv'
data_DM = xgb.DMatrix('{0}?format=csv&label_column=0'.format(rfname), label=1,
                      feature_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'])
print(data_DM)
print(data_DM.feature_names)
print(data_DM.feature_types)
print(data_DM.get_base_margin())
print(data_DM.num_col())
print(data_DM.num_row())
print(data_DM.get_label())
print(data_DM.get_weight())
#print(data_DM.get_float_info())
'''
'''
data = reader(rfname)
len_data = len(data)
len_col = 15
# feature(x_data)와 result(y_data) 나누기
x_data = np.array([list(map(int, [e for e in elem[:len_col]])) for elem in data[:len_data]])
y_data = np.array( [ [int( elem[len_col] ) for elem in data[:len_data]] ] )

x_data = xgb.DMatrix(x_data)
y_data = xgb.DMatrix(y_data)
print(x_data)
print(y_data)
#bst = xgb.train(param, x_data, num_round)
'''

data = np.loadtxt('testdata_2.csv', delimiter=',')
len_data = 10000
x_data = np.array([list(map(int, [e for e in elem[:15]] )) for elem in data[:len_data]])
y_data = np.array([int(elem[15]) for elem in data[:len_data]])

x_train, y_train, = x_data[:len(x_data):2], y_data[:len(y_data):2]
x_test, y_test = x_data[1:len(x_data):2], y_data[1:len(y_data):2]

model = xgb.XGBClassifier()
model.fit(x_train, y_train)
pre = model.predict(x_test)
print(pre)

xgb.plot_importance(model)
plt.show()

params = {
    'colsample_bynode' : 0.8,
    'learning_rate' : 1,
    'max_depth' : 5,
    'num_parallel_tree' : 100,
    'objective' : 'binary:logistic',
    'subsampling' : 0.8,
    'tree_method' : 'gpu_hist'
}
bst = xgb.train(params, dmatrix, num_boost_round = 1)
