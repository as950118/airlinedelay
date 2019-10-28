import pandas as pd
#reader = lambda rfname: pd.get_dummies(pd.read_csv(rfname, delimiter=',', header=0))
import csv
reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))

df = reader('weather_2.csv')
new_df = [[] for i in range(len(df))]
new_df[0].append(df[0][1])
new_df[0] += df[0][3:]
print(new_df[0])
for i in range(1, len(df)):
    y = df[i][0][:4]
    m = df[i][0][4:6]
    d = df[i][0][6:8]
    h = df[i][0][8:]
    new_df[i].append(int(y))
    new_df[i].append(int(m))
    new_df[i].append(int(d))
    new_df[i].append(int(h))
    new_df[i].append(df[i][1])
    new_df[i] += df[i][3:]
#print(new_df)
from collections import defaultdict
count = defaultdict(lambda : [0 for i in range(31)])
weather = defaultdict(lambda :[[[0 for i in range(10)] for j in range(24)] for k in range(31)])
for i in range(len(new_df)):
    if new_df[i][1] == 9 and new_df[i][2] > 15:
        for j in range(10):
            weather[new_df[i][4]][new_df[i][2]][new_df[i][3]][j] = weather[new_df[i][4]][new_df[i][2]][new_df[i][3]][j] + float(new_df[i][j+5])
            if count[new_df[i][4]][new_df[i][2]]:
                weather[new_df[i][4]][new_df[i][2]][new_df[i][3]][j] /= 2
            else:
                count[new_df[i][4]][new_df[i][2]] += 1
#weather = [new_df[i] for i in range(len(new_df)) if new_df[i][1]==9 and new_df[i][2]>15]
out = []
for arp in weather:
    for day, val in enumerate(weather[arp]):
        if val[0][0]:
            for hour, elem in enumerate(weather[arp][day]):
                out.append([2019, 9, day, hour, arp] + elem)

#out = [[weather[key][i], key] + weather[key] for key in weather for j in range() if weather[key][i][0] != 0]
print(out)
wf = writer('weather_16_30_2.csv')
for i in out:
    wf.writerow(i)