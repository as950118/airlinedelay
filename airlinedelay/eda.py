#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
datas = pd.read_csv("AFSNT.csv", engine='python',encoding='cp949')
SDT_MM_DD = datas[['SDT_MM', 'SDT_DD']].values
print(SDT_MM_DD, type(SDT_MM_DD))
'''
test = ['N','Y','N','Y']
test = [1 if i=='N' else 0 for i in test]
print(test)
datas = pd.read_csv("AFSNT.csv", engine='python',encoding='cp949')
SDT_MM_DD = datas[['SDT_MM', 'SDT_DD']].as_matrix()
DLY = datas[['DLY']].as_matrix()
DLY = [1 for i in DLY if i=='N']
for i in range(len(DLY)):
    print(DLY[i])
'''
