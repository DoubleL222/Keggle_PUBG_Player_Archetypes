import pandas as pd

#nrows=20
#usecols=[1,2]
Agg_Data = pd.read_csv('Data/agg_match_stats_0.csv', error_bad_lines=False)

'''
for x in range(1,5):
    print("Doing file #"+str(x))
    new_data = pd.read_csv('Data/agg_match_stats_'+str(x)+'.csv', error_bad_lines=False)
    Agg_Data = Agg_Data.append(new_data, ignore_index=True)
    #Death_Data = Death_Data.append(pd.read_csv('# FILENAME'+str(x)+'.csv', nrows=50000, error_bad_lines=False))
'''




print(Agg_Data[:, 1])

#Agg_Data.player_name.value_count()