import pandas as pd
#reader = lambda rfname: pd.get_dummies(pd.read_csv(rfname, delimiter=',', header=0))
import csv
reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))

df = reader('weather_2.csv')
new_df = [[] for i in range(15)]