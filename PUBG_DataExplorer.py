import pandas as pd

#usecols determine which column you are importing
#nrows=20 will give you the first 20 rows.
data = pd.read_csv('Data/agg_match_stats_0.csv', usecols=[1,2], error_bad_lines=False)

#instead, just append it by the different csv files.
combinedData = data.append(data)

#When done, then length should be around 65 million
print(len(combinedData))