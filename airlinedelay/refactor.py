import csv
from collections import defaultdict
reader = lambda rname:list(csv.reader(open(rname), delimiter=','))
writer = lambda wname:csv.writer(open(wname, 'w', newline=''))

afsnt = reader("./AFSNT.csv")
weather = reader("./weather_2.csv")
ydmt = defaultdict()
print("AFSNT :", afsnt[0:2]) #15
print("weather :", weather[0:2]) #5
def func(i, temp):
    try:
        t = str(int(afsnt[i][0])-temp)
    except:
        return 0
    try:
        afsnt[i] += ydmt[t + afsnt[i][6]]
        return 1
    except:
        try:
            t = str(int(afsnt[i][0])+temp)
        except:
            return 0
        try:
            afsnt[i] += ydmt[t + afsnt[i][6]]
            return 1
        except:
            return 0

for i in range(len(weather)):
    ydmt[weather[i][0]+weather[i][1]]=weather[i][3:]
for i in range(len(afsnt)):
    try:
    #if ydmt[afsnt[i][0]+afsnt[i][6]]:
        afsnt[i] += ydmt[afsnt[i][0]+afsnt[i][6]]
    except:
    #else:
        temp = 1
        print(i, temp)
        while not func(i, temp):
            print(temp)
            temp+=1
wf = writer("AFSNT_weather_4.csv")
for i in range(len(afsnt)):
    wf.writerow(afsnt[i])