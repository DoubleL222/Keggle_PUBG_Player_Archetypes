import pandas as pd
import numpy
import Measures as measures

temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', error_bad_lines=False)
'''
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_1.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_2.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_3.csv', error_bad_lines=False))
temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_4.csv', error_bad_lines=False))

temp_agg_data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)
temp_agg_data.append(pd.read_csv('output_data/summary_data_2000.csv', error_bad_lines=False))
temp_agg_data.append(pd.read_csv('output_data/summary_data_3000.csv', error_bad_lines=False))
unique_weapons = temp_agg_data['weapon_usage'].unique()
unique_deaths = temp_agg_data['killed_by'].unique()

uniques = set()
for _val in unique_deaths:
    uniques.add(_val)
for _val in unique_weapons:
    uniques.add(_val)

print("uniques: ", uniques)
'''
print('Finished reading')
print(len(temp_death_data.index))
allWeapons = set()
for index, row in temp_death_data.iterrows():
    allWeapons.add(measures.get_weapon_category(row['killed_by'].lower()))
    if index % 100 == 0:
        print(index)

print(allWeapons)

