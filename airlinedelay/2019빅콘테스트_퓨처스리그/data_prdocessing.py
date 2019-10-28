import pandas as pd
import csv
import numpy as np
import math
from collections import defaultdict
import datetime as dt
#writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))
reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr')
#writer = lambda wfname: pd.to_csv(wfname, encoding='euc-kr')

class DataProcess:
    def __init__(self, rfname, rfname_weather):
        self.rfname = rfname
        self.rfname_weather = rfname_weather

        self.dataset = reader(rfname)
        self.dataset.drop('DRR', axis=1, inplace=True) #row면 0, col이면 1
        self.dataset.drop('CNR', axis=1, inplace=True)
        self.feature_names = self.dataset.columns.values.tolist()
        self.data = self.dataset.get_values().tolist()

        self.dataset_weather = reader(rfname_weather)
        self.feature_names_weather = self.dataset_weather.columns.values.tolist()
        self.data_weather = self.dataset_weather.get_values().tolist()
    def refactor_AFSNT(self):
        def func(cur, temp, idx):
            try:
                if self.data_weather[cur - temp][idx]:
                    self.data_weather[cur][idx] = self.data_weather[cur - temp][idx]
                return 1
            except:
                try:
                    if self.data_weather[cur + temp][idx]:
                        self.data_weather[cur][idx] = self.data_weather[cur + temp][idx]
                    return 1
                except:
                    return 0

        #week_list = ['일','월','화','수','목','금','토']
        #FLO_list = ['J','L','A','B','F','H','I']
        idx2char = defaultdict()
        char2idx = defaultdict()
        for feature in self.feature_names:
            idx2char[feature] = list(set(self.dataset[feature].get_values()))
            char2idx[feature] = {c: i for i, c in enumerate(idx2char[feature])}
        self.rm_rows = []
        for cur in range(len(self.data)):
            #CNL 행 제거
            if self.data[cur][14] == 'Y':
                self.rm_rows.append(cur)
                continue
            #요일을 숫자로
            self.data[cur][3] = char2idx['SDT_DY'][self.data[cur][3]]
            #ARP, DOP 숫자로
            self.data[cur][4] = self.data[cur][4][3:]
            self.data[cur][5] = self.data[cur][5][3:]
            #FLO 숫자로
            self.data[cur][6] = char2idx['FLO'][self.data[cur][6]]
            #FLT 숫자로(항공사 구분)
            self.data[cur][7] = str(char2idx['FLO'][self.data[cur][7][0]]) + str(self.data[cur][7][1:])
            try:
                int(self.data[cur][7][-1])
            except:
                self.data[cur][7] = self.data[cur][7][:-1]
            #REG도 숫자로?
            self.data[cur][8] = char2idx['REG'][self.data[cur][8]]
            #AOD 숫자로?
            self.data[cur][9] = char2idx['AOD'][self.data[cur][9]]
            #IRR?
            self.data[cur][10] = char2idx['IRR'][self.data[cur][10]]
            #STT 숫자로
            t,m = map(int, self.data[cur][11].split(':'))
            self.data[cur][11] = t*60 + m
            #ATT 숫자로
            t,m = map(int, self.data[cur][12].split(':'))
            self.data[cur][12] = t*60 + m
            #DLY 숫자로
            self.data[cur][13] = char2idx['DLY'][self.data[cur][13]]
            #STTATT(실제출발-예상출발) 구하기
            self.data[cur].append(self.data[cur][12]-self.data[cur][11])

        self.feature_names.append('STTATT')
        # CNL 행 삭제
        for i, row in enumerate(self.rm_rows):
            self.data.pop(row-i)
            #self.data.drop(self.data, (row), axis=0)


    def refactor_weather(self):
        def func(cur, temp, idx):
            try:
                if str(self.data_weather[cur - temp][idx]) != 'nan':
                    self.data_weather[cur][idx] = self.data_weather[cur - temp][idx]
                return 1
            except:
                try:
                    if str(self.data_weather[cur + temp][idx]) != 'nan':
                        self.data_weather[cur][idx] = self.data_weather[cur + temp][idx]
                    return 1
                except:
                    return 0

        necessary = ['기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '현지기압(hPa)']
        Notnecessary = ['강수량(mm)', '일조(hr)', '적설(cm)', '전운량(10분위)', '최저운고(100m )']
        #print(self.data_weather[0])
        #print(self.data_weather[0][6],type(self.data_weather[0][6]),str(self.data_weather[0][6]),float(self.data_weather[0][6]))
        #print(str(self.data_weather[0][6]) == 'nan')
        #input()
        for cur in range(len(self.data_weather)):
            for n in necessary:
                idx = self.feature_names_weather.index(n)
                if str(self.data_weather[cur][idx]) == 'nan':
                    temp = 1
                    while not func(cur, temp, idx):
                        temp += 1
            for nn in Notnecessary:
                idx = self.feature_names_weather.index(nn)
                if str(self.data_weather[cur][idx]) =='nan':
                    self.data_weather[cur][idx] = 0

    def merge(self):
        data_ydmt = [str(d[0]) + str(d[1]).rjust(2, "0") + str(d[2]).rjust(2, "0") + str(d[11]//60).rjust(2, "0") for d in self.data]
        ydmt = defaultdict()
        def func(i, temp):
            print(i,temp)
            try:
                t = str(int(data_ydmt[i]) - temp)
            except:
                return 0
            try:
                self.data[i] = self.data[i] + ydmt[t + self.data[i][4]]
                return 1
            except:
                try:
                    t = str(int(data_ydmt[i]) + temp)
                except:
                    return 0
                try:
                    self.data[i] = self.data[i] + ydmt[t + self.data[i][4]]
                    return 1
                except:
                    return 0

        for d in self.data_weather:
            ydmt[str(d[0])+str(d[1]).rjust(2, "0")+str(d[2]).rjust(2, "0")+str(d[3]).rjust(2, "0") + d[4][3:]] = d[5:]
        for i in range(len(self.data)):
            try:
                self.data[i] = self.data[i] + ydmt[str(data_ydmt[i]) + self.data[i][4]]
            except:
                temp = 1
                while not func(i, temp):
                    temp += 1

    def drop(self):
        drop_list = [('SDT_YY', 1), ('CNL', 1), ('REG', 1), ('IRR', 1), ('ATT', 1), ('STTATT', 1)]
        for drop, axis in drop_list:
            self.df.drop(drop, axis=axis, inplace=True)

    def out(self, wfname):
        self.df.to_csv(wfname, encoding='euc-kr', index=0)

    def run(self):
        print("**********************************")
        print("**********Refactor AFSNT**********")
        print("**********************************")
        self.refactor_AFSNT()
        print("************************************")
        print("**********Refactor weahter**********")
        print("************************************")
        self.refactor_weather()
        print("*************************")
        print("**********Merge**********")
        print("*************************")
        self.merge()
        self.df = pd.DataFrame(self.data, columns=self.feature_names + self.feature_names_weather[5:])
        print("*************************")
        print("**********Drop***********")
        print("*************************")
        self.drop()
        print("*************************")
        print("**********Out************")
        print("*************************")
        self.out("merge_AFSNT_weather.csv")

if __name__ == "__main__":
    DataProcess('AFSNT.csv', 'weather.csv').run()