import csv
from collections import defaultdict
reader = lambda rname:list(csv.reader(open(rname), delimiter=','))
writer = lambda wname:csv.writer(open(wname, 'w', newline=''))

weather = reader("./weather.csv")

def func(cur, temp, idx):
    try:
        if weather[cur-temp][idx]:
            weather[cur][idx] = weather[cur-temp][idx]
        return 1
    except:
        try:
            if weather[cur + temp][idx]:
                weather[cur][idx] = weather[cur + temp][idx]
            return 1
        except:
            return 0

necessary = [3,5,6,7,8]
Notnecessary = [4,9,10,11,12]

for cur in range(len(weather)):
    for idx in necessary:
        if not weather[cur][idx]:
            temp = 1
            print(cur, temp)
            while not func(cur, temp, idx):
                temp+=1
    for idx in Notnecessary:
        if not weather[cur][idx]:
            weather[cur][idx] = 0

wf = writer("weather_2.csv")
for i in range(len(weather)):
    wf.writerow(weather[i])