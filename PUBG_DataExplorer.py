import pandas as pd
import Measures as measure
import numpy as np
import time


def Write_To_File(_i, data):
    pandas_summary_data = pd.DataFrame(data)
    # PROPERLY RENAME COLUMNS
    pandas_summary_data.columns = ['player_name', 'match_id', 'distance_walked', 'distance_rode', 'travel_ratio',
                                   'kill_count', 'knockdown_count', 'player_assists', 'kill_knockdown_ratio',
                                   'kill_distance',
                                   'survive_time', 'player_dmg', 'weapon_usage', 'killed_by', 'killed_from',
                                   'team_placement', 'match_mode', 'party_size', 'game_size', 'match_mode',
                                   'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun',
                                   'Pistols and Sidearm',
                                   'Melee', 'Crossbow', 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
                                   ]
    # SAVE MATRIX TO FILE
    # print(list(pandas_summary_data.columns.values))
    pandas_summary_data.to_csv("output_data/summary_data_" + str(_i) + ".csv", index=False)


# Number of matches wanted
number_of_matches = 10000

different_weapons = [x.lower() for x in ['Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm',
                     'Melee', 'Crossbow', 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out']]

# LOADING IN SOME DATA
'''
temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', nrows=5000, error_bad_lines=False)
temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', nrows=5000, error_bad_lines=False)
'''

# '''
temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', error_bad_lines=False)
temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', error_bad_lines=False)
# '''

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

# print(list(temp_agg_data.columns.values))
# print(list(temp_death_data.columns.values))
temp_agg_data.sort_values(by=['match_id'], inplace=True)
# temp_death_data.sort_values(by=['match_id'], inplace=True)

match_summary_data = []

start_time = time.time()

prevMatchId = ""

print("STATUS: finished pre-processing")

i = 0
match_count = 0
'''
unique_matches = temp_agg_data['match_id'].unique()
print("unique matches: ", unique_matches.__len__())
selected_unique_matches = unique_matches[0:number_of_matches]
selected_death_data = temp_death_data.loc[temp_death_data['match_id'].isin(selected_unique_matches)]
print("selected_death_data of deaths: ", len(selected_death_data.index))
selected_agg_data = temp_agg_data.loc[temp_agg_data['match_id'].isin(selected_unique_matches)]
print("Number of agg data: ", len(selected_agg_data.index))
print("old agg data: ",len(temp_agg_data.index))
temp_agg_data = selected_agg_data
print("NEW agg data: ",len(temp_agg_data.index))
print("old death data: ",len(temp_death_data.index))
temp_death_data = selected_death_data
print("NEW agg data: ",len(temp_death_data.index))
'''
last_match_start_time = time.time()
for index, row in temp_agg_data.iterrows():
    # print("Index: ", index, "; time: ", time.time() - start_time)
    '''
    if i % 10000 == 0:
        print('Rows: ', i, '; time:',format(time.time()-start_time, '.2f'))
    '''
    curr_match_id = row['match_id']
    if not prevMatchId == curr_match_id:
        start_reading_deaths = time.time()
        '''
        if not i == 0:
            # print("Dropping Data")
            temp_death_data.drop(temp_death_data.index[[death_data.index.values]], inplace=True)
            temp_death_data.reset_index(drop=True, inplace=True)
        '''
        death_data = temp_death_data.loc[temp_death_data['match_id'] == curr_match_id]
        # temp_death_data.drop(temp_death_data.index[[death_data.index.values]], inplace=True)
        # temp_death_data.reset_index(drop=True, inplace=True)
        prevMatchId = curr_match_id
        match_count += 1
        # print('next match; at row ', i, '; Time it took to read deaths: ',time.time()-start_reading_deaths)
        if match_count % 1000 == 0:
            print('writing file: ', match_count, '; took: ', format(time.time() - last_match_start_time, '.2f'), " s")
            Write_To_File(match_count, match_summary_data)
            match_summary_data = []
            last_match_start_time = time.time()

    # Find all rows for the current match
    # death_data = temp_death_data.loc[temp_death_data['match_id'] == row['match_id']]

    # Find row equal to player's death (may be null)
    # start_reading_row = time.time()
    death = death_data.loc[death_data['victim_name'] == row['player_name']]

    # Find rows equal to players kills
    kills = death_data.loc[death_data['killer_name'] == row['player_name']]

    # Make weapons array
    weapons_columns = np.zeros(len(different_weapons))
    wep = measure.weapon_ratio(kills)
    if not wep == 'none':
        weapons_columns[different_weapons.index(wep)] = 1

    new_row = [row['player_name'], row['match_id'], row['player_dist_walk'], row['player_dist_ride'],
         measure.travel_ratio(row), row['player_kills'], row['player_dbno'], row['player_assists'],
         measure.kill_knockdown_ratio(row), measure.kill_distance(kills), row['player_survive_time'], row['player_dmg'],
         measure.weapon_ratio(kills), measure.weapon_ratio(death), measure.kill_distance(death),
         row['team_placement'], row['match_mode'], row['party_size'], row['game_size'], row['match_mode']]
    new_row.extend(weapons_columns)
    # Append summarized data
    match_summary_data.append(new_row)
    i += 1

elapsed_time = time.time() - start_time
print("ELAPSED TIME: ", elapsed_time)
print("STATUS: finished _summation")

# CONVERTE TO PANDAS MATRIX
pandas_summary_data = pd.DataFrame(match_summary_data)
# PROPERLY RENAME COLUMNS
pandas_summary_data.columns = ['player_name', 'match_id', 'distance_walked', 'distance_rode', 'travel_ratio',
                               'kill_count', 'knockdown_count', 'player_assists', 'kill_knockdown_ratio',
                               'kill_distance',
                               'survive_time', 'player_dmg', 'weapon_usage', 'killed_by', 'killed_from',
                               'team_placement', 'match_mode', 'party_size', 'game_size', 'match_mode',
                               'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun',
                               'Pistols and Sidearm',
                               'Melee', 'Crossbow', 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
                               ]
# SAVE MATRIX TO FILE
pandas_summary_data.to_csv("summary_data.csv", index=False)
