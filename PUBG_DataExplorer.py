import pandas as pd

#nrows=20
#usecols=[1,2]
temp_agg_data = pd.read_csv('agg_match_stats_0.csv', nrows=50000, error_bad_lines=False)
temp_death_data = pd.read_csv('agg_match_stats_0.csv', nrows=50000, error_bad_lines=False)

for x in range(1,5):
    print("Doing file #"+str(x))
    Agg_Data = Agg_Data.append(pd.read_csv('agg_match_stats_'+str(x)+'.csv', nrows=50000, error_bad_lines=False))
    #Death_Data = Death_Data.append(pd.read_csv('FILENAME'+str(x)+'.csv', nrows=50000, error_bad_lines=False))


#Agg_Data.player_name.value_count()