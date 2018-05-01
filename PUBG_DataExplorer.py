import pandas as pd
import Measures as measure
import time


# nrows=20
# usecols=[1,2]

#LOADING IN SOME DATA
#'''
temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', nrows=5000, error_bad_lines=False)
temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', nrows=5000, error_bad_lines=False)
#'''

'''
#LOADING IN ALL DATA
temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', error_bad_lines=False)
temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_1.csv', error_bad_lines=False))
temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_2.csv', error_bad_lines=False))
temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_3.csv', error_bad_lines=False))
temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_4.csv', error_bad_lines=False))

temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', error_bad_lines=False)
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_1.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_2.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_3.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_4.csv', error_bad_lines=False))
'''
temp_agg_data.sort_values(by=['match_id'], inplace=True)
#temp_death_data.sort_values(by=['match_id'], inplace=True)

match_summary_data = []

start_time = time.time()

prevMatchId = ""

i = 0
for index, row in temp_agg_data.iterrows():
    #print("Index: ", index, "; time: ", time.time() - start_time)
    if i % 1000 == 0:
        print('Rows: ', i, '; time:',format(time.time()-start_time, '.2f'))
    curr_match_id = row['match_id']
    if not prevMatchId == curr_match_id:
        death_data = temp_death_data.loc[temp_death_data['match_id'] == curr_match_id]
        temp_death_data.drop(temp_death_data.index[[death_data.index.values]], inplace=True)
        temp_death_data.reset_index(drop=True, inplace=True)
        prevMatchId = curr_match_id

    # Find all rows for the current match
    death_data = temp_death_data.loc[temp_death_data['match_id'] == row['match_id']]

    # Find row equal to player's death (may be null)
    death = death_data.loc[temp_death_data['victim_name'] == row['player_name']]

    # Find rows equal to players kills
    kills = death_data.loc[temp_death_data['killer_name'] == row['player_name']]

    # Append summarized data
    match_summary_data.append(
        [row['player_name'], row['match_id'], measure.travel_ratio(row), measure.kill_knockdown_ratio(row),
         measure.kill_distance(row, kills), row['player_survive_time'], row['player_dmg'],
         measure.weapon_ratio(row, kills)])
    i+=1

elapsed_time = time.time() - start_time
print("ELAPSED TIME: ", elapsed_time)

#CONVERTE TO PANDAS MATRIX
pandas_summary_data = pd.DataFrame(match_summary_data)
#PROPERLY RENAME COLUMNS
pandas_summary_data.columns = ['player_name', 'match_id', 'travel_ratio', 'kill_knockdown_ratio', 'kill_dsitance', 'survive_time', 'player_dmg', 'weapon_usage']
#SAVE MATRIX TO FILE
pandas_summary_data.to_csv("summary_data.csv")
