import pandas as pd
import numpy as np
import hdbscan
import seaborn as sns

import matplotlib.pyplot as pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

# # nrows=20
# # usecols=[1,2]
#
# #LOADING IN SOME DATA
# '''
# temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', nrows=1000, error_bad_lines=False)
# temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', nrows=1000, error_bad_lines=False)
# '''
#
# #'''
# #LOADING IN ALL DATA
# temp_agg_data = pd.read_csv('data/aggregates/agg_match_stats_0.csv', error_bad_lines=False)
# temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_1.csv', error_bad_lines=False))
# temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_2.csv', error_bad_lines=False))
# temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_3.csv', error_bad_lines=False))
# temp_agg_data.append(pd.read_csv('data/aggregates/agg_match_stats_4.csv', error_bad_lines=False))
#
# temp_death_data = pd.read_csv('data/deaths/kill_match_stats_final_0.csv', error_bad_lines=False)
# temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_1.csv', error_bad_lines=False))
# temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_2.csv', error_bad_lines=False))
# temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_3.csv', error_bad_lines=False))
# temp_death_data.append(pd.read_csv('data/deaths/kill_match_stats_final_4.csv', error_bad_lines=False))
# #'''
# match_summary_data = []
#
# for index, row in temp_agg_data.iterrows():
#     # Find all rows for the current match
#     death_data = temp_death_data.loc[temp_death_data['match_id'] == row['match_id']]
#
#     # Find row equal to player's death (may be null)
#     death = death_data.loc[temp_death_data['victim_name'] == row['player_name']]
#
#     # Find rows equal to players kills
#     kills = death_data.loc[temp_death_data['killer_name'] == row['player_name']]
#
#     # Append summarized data
#     match_summary_data.append(
#         [row['player_name'], row['match_id'], measure.travel_ratio(row), measure.kill_knockdown_ratio(row),
#          measure.kill_distance(row, kills), row['player_survive_time'], row['player_dmg'],
#          measure.weapon_ratio(row, kills)])
#
# #CONVERTE TO PANDAS MATRIX
# pandas_summary_data = pd.DataFrame(match_summary_data)
# #PROPERLY RENAME COLUMNS
# pandas_summary_data.columns = ['player_name', 'match_id', 'travel_ratio', 'kill_knockdown_ratio', 'kill_dsitance', 'survive_time', 'player_dmg', 'weapon_usage']
# #SAVE MATRIX TO FILE
# pandas_summary_data.to_csv("summary_data.csv")

# NORMALISE COLUMN (value to range [0,1])
def normalise_column(_col):
    max_val = np.amax(_col)
    min_val = np.amin(_col)
    new_col = []
    for _val in _col:
        _newVal = (_val - min_val) / (max_val - min_val)
        new_col.append(_newVal)
    return np.array(new_col)


# FIND OUTLIERS USING Quartiles
def find_outliers(_col, outer_fence_factor):
    q1 = _col.quantile(0.25)
    q3 = _col.quantile(0.75)
    q_i_r = q3 - q1
    # Extended outer fences to not include 38 years old
    outer_fence_low = q1 - q_i_r * outer_fence_factor
    outer_fence_high = q3 + q_i_r * outer_fence_factor
    indices = []
    for _val in _col:
        if _val < outer_fence_low or _val > outer_fence_high:
            indices.append(list(_col).index(_val))
            # print(_val)
    print("Indices: " + str(indices.__len__()))
    return indices


# REPLACE OUTLIERS IN _outlier_indices WITH MEDIAN COLUMN VALUE
def replace_outliers_with_medians(_data, _colName, _outlier_indice):
    median_value = _data[_colName].median()
    _data.loc[_colName, _outlier_indice] = median_value
    return


if __name__ == '__main__':
    pyplot.close('all')
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    pca = PCA(n_components=2)
    tSne = TSNE(n_components=2, init='pca', n_iter=1000, n_iter_without_progress=300, verbose=4)

    data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)

    # for i in range(2, 30):
    #     data = data.append(pd.read_csv('output_data/summary_data_' + str(i) + '0.csv', error_bad_lines=False))

    data.reset_index(inplace=True)

    print("Data Loaded!")

    print("Removing players with no kills...")
    # Delete the rows with label "Ireland"
    # For label-based deletion, set the index first on the dataframe:
    data = data.query('kill_count != 0')
    #data = data.dropna(how='all')
    data.reset_index(inplace=True)
    print("Number of rows after cleaning: " + str(data.__len__()))

    # Clean Data
    print("Maximum kill distance: " + str(data['kill_distance'].max()))

    print("Normalizing data...")
    # Normalize Data
    data['distance_walked'] = normalise_column(data['distance_walked'])
    data['distance_rode'] = normalise_column(data['distance_rode'])
    data['kill_count'] = normalise_column(data['kill_count'])
    data['player_assists'] = normalise_column(data['player_assists'])
    data['knockdown_count'] = normalise_column(data['knockdown_count'])
    data['kill_distance'] = normalise_column(data['kill_distance'])
    data['survive_time'] = normalise_column(data['survive_time'])
    data['player_dmg'] = normalise_column(data['player_dmg'])
    data['killed_from'] = normalise_column(data['killed_from'])
    data['team_placement'] = normalise_column(data['team_placement'])
    data['party_size'] = normalise_column(data['party_size'])

    selected_data = data[[
        'distance_walked', 'distance_rode',
        'travel_ratio', 'kill_count',
        'knockdown_count', 'player_assists',
        'kill_knockdown_ratio', 'kill_distance',
        'survive_time', 'player_dmg', 'team_placement'
        #,'party_size'
        #,'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm', 'Melee', 'Crossbow'
        #,'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
    ]]
    significance = 0.03
    print("Fitting and transforming data...")
    selected_data = pd.DataFrame(tSne.fit_transform(selected_data))
    selected_data.to_csv(path_or_buf="TSNE-fitted data without weapons and party_size.csv", index=False)
    #selected_data = pd.read_csv(filepath_or_buffer="TSNE-fitted data without weapons and party_size.csv")
    print("Running HDBSCAN...")
    hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=int(data.__len__()*significance), alpha=1.0)
    hdbscan_instance.fit(selected_data)

    print("# of HDBSCAN labels: " + str(hdbscan_instance.labels_.max()+1))
    print("HDBSCAN done!")

    # print("Plotting condensed tree...")
    # hdbscan_instance.condensed_tree_.plot()
    # pyplot.show()

    # Draw scatter plots
    print('Drawing scatter plots...')
    #dist_mat = generate_distance_matrix(clean_data)
    # transformed = pd.DataFrame(pca.fit_transform(selected_data))

    plot_kwds = {'alpha': 0.4, 's': 1, 'linewidths': 0}

    palette = sns.color_palette('hls', hdbscan_instance.labels_.max()+1)
    cluster_colors = [sns.desaturate(palette[col], np.clip(sat*2, 0.0, 1.0))
                      if col >= 0 else (0.8, 0.8, 0.8) for col, sat in
                      zip(hdbscan_instance.labels_, hdbscan_instance.probabilities_)]

    pyplot.scatter(selected_data[0], selected_data[1], color=cluster_colors, **plot_kwds)
    pyplot.title(str(hdbscan_instance.min_samples))
    pyplot.savefig('scatterplot' + str(hdbscan_instance.min_samples) + '.png', dpi=300)
    pyplot.show()
