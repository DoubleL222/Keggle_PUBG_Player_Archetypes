import pandas as pd
import Measures as measure

# nrows=20
# usecols=[1,2]
temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', nrows=50000, error_bad_lines=False)
temp_death_data = pd.read_csv('data/deaths/agg_match_stats_0.csv', nrows=50000, error_bad_lines=False)

'''
for x in range(1,5):
    print("Doing file #"+str(x))
    Agg_Data = Agg_Data.append(pd.read_csv('agg_match_stats_'+str(x)+'.csv', nrows=50000, error_bad_lines=False))
    #Death_Data = Death_Data.append(pd.read_csv('FILENAME'+str(x)+'.csv', nrows=50000, error_bad_lines=False))
'''
match_summary_data = [];

for index, row in temp_agg_data.iterrows():
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

print(match_summary_data[0])
