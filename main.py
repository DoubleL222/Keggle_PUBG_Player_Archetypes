import numpy as np

allData = []
for line in open('Data/agg_match_stats_0.csv'):
    newData = line.split()
    addData = []
    for entry in newData:
        addData.append(entry.strip().strip())
    allData.append(addData)