import pandas as pd
import csv
from collections import defaultdict
import datetime as dt
reader = lambda rfname: pd.read_csv(rfname, delimiter=',', header=0, encoding='euc-kr'))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))

class DataProcess:
    def __init__(self, rfname, rfname_weather):
        self.rfname = rfname
        self.rfname_weather = rfname_weather

        self.dataset = reader(rfname)
        self.feature_names = self.dataset.columns.values
        self.data = self.dataset.get_values()

        self.dataset_weather = reader(rfname_weather)
        self.feature_names_weather = self.dataset_weather.columns.values
        self.data_weather = self.dataset_weather.get_values()

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
            idx2char[feature] = list(set(self.data[feature].get_values()))
            char2idx[feature] = {c: i for i, c in enumerate(idx2char[feature])}
        rm_rows = []
        for cur in range(len(self.data)):
            #CNL 행 제거
            if self.data[cur][15] == 'Y':
                rm_rows.append(cur)
                continue
            #요일을 숫자로
            self.data[cur][3] = char2idx['SDT_DY'].index(self.data[cur][3])
            #ARP, DOP 숫자로
            self.data[cur][4] = self.data[cur][4][3:]
            self.data[cur][5] = self.data[cur][5][3:]
            #FLO 숫자로
            self.data[cur][6] = char2idx['FLO'].index(self.data[cur][6])
            #FLT 숫자로(항공사 구분)
            self.data[cur][7] = str(char2idx['FLO'].index(self.data[cur][7][1])) + str(self.data[cur][7][1:])
            #REG도 숫자로?
            self.data[cur][8] = char2idx['REG'].index(self.data[cur][8])
            #AOD 숫자로?
            self.data[cur][9] = char2idx['AOD'].index(self.data[cur][9])
            #IRR?
            self.data[cur][10] = char2idx['IRR'].index(self.data[cur][10])
            #STT 숫자로
            t,m = map(int, self.data[cur][11].split(':'))
            self.data[cur][11] = t*60 + m
            #ATT 숫자로
            t,m = map(int, self.data[cur][12].split(':'))
            self.data[cur][12] = t*60 + m
            #DLY 숫자로
            self.data[cur][13] = char2idx['DLY'].index(self.data[cur][13])
            #DRR 숫자로
            self.data[cur][14] = char2idx['DLY'].index(self.data[cur][14])

        #CNL 행 삭제제
        for row in rm_rows:
            self.data.pop(row)

    def refactor_weather(self):
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

        necessary = ['기온', '풍속(m/s)', '풍향(16방위)', '습도(%)', '현지기압(hPa)']
        Notnecessary = ['강수량(mm)', '일조(hr)', '적설(cm)', '전운량(10분위)', '최저운고(100m )']

        for cur in range(len(self.data_weather)):
            for idx, n in enumerate(necessary):
                if not self.data_weather[cur][idx]:
                    temp = 1
                    print(cur, temp)
                    while not func(cur, temp, idx):
                        temp += 1
            for idx, nn in enumerate(Notnecessary):
                if not self.data_weather[cur][idx]:
                    self.data_weather[cur][idx] = 0

    def merge(self):
        data_ydmt = [str(d[0])+str(d[1])+str(d[2])+str( list(map(str, d[14].split(':')))[0] ) for d in self.data]
        ydmt = defaultdict()
        def func(i, temp):
            try:
                t = str(int(data_ydmt[i]) - temp)
            except:
                return 0
            try:
                self.data[i] += ydmt[t + self.data[i][4]]
                return 1
            except:
                try:
                    t = str(int(data_ydmt[i]) + temp)
                except:
                    return 0
                try:
                    self.data[i] += ydmt[t + self.data[i][4]]
                    return 1
                except:
                    return 0

        for i in range(len(self.data_weather)):
            ydmt[self.data_weather[i][0] + self.data_weather[i][1]] = self.data_weather[i][3:]
        for i in range(len(self.data)):
            try:
                self.data[i] += ydmt[data_ydmt[i] + self.data[i][4]]
            except:
                temp = 1
                while not func(i, temp):
                    temp += 1

    def out(self, wfname):
        wf = writer(wfname)
        for i in range(len(self.data)):
            wf.writerow(self.data[i])

    def run(self):
        self.refactor_AFSNT()
        self.refactor_weather()
        self.merge()
        self.out("merge_AFSNT_weather.csv")

if __name__ == "__main__":
    DataProcess('AFSNT.csv', 'weather.csv').run()