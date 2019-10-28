import tensorflow as tf
import numpy as np
import pandas as pd

#Rawdata
rfname = "AFSNT_weather_2.csv"
encoding = 'ecu-kr'
names = ['SDT_MM', 'SDT_DD', 'SDT_DY','ARP_N','ODP_N','기온','습도','현지기압']
raw_dataframe = pd.read_csv(rfname, names=names, encoding=encoding)
print(raw_dataframe.info())
datas = raw_dataframe.values[:].astype(np.float)

#For HyperParameter Tuning
tf.set_random_seed(777)

#Standarization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

def min_max_scailing(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

def reverse_min_max_scailing(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

#HyperParameter
input_data_colnum_cnt = len(datas[0])
output_data_colunum_cnt = 1

seq_length = 28
rnn_cell_hidden_dim = 20
forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 1.0

epoch_num = 1000
learning_rate = 0.01

SDT = datas[:, 2:5]
SDT = min_max_scailing(SDT)

ARP = datas[:, ]
ARP = min_max_scailing(ARP)
WTH = datas[:]
WTH = min_max_scailing(WTH)

datax = []
datay = []

for i in range():