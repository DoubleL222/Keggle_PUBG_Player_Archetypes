import pandas as pd
import numpy as np
import hdbscan
import seaborn as sns
import collections

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
def normalise_column(_col, _multiplier=1.0):
    max_val = np.amax(_col)
    min_val = np.amin(_col)
    new_col = []
    for _val in _col:
        _newVal = ((_val - min_val) / (max_val - min_val)) * _multiplier
        new_col.append(_newVal)
    return np.array(new_col)


def multiply_column(_col, _multiplier):
    new_col = []
    for _val in _col:
        new_col.append(_val*_multiplier)
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
    #print("Indices: " + str(indices.__len__()))
    return indices


# REPLACE OUTLIERS IN _outlier_indices WITH MEDIAN COLUMN VALUE
def replace_outliers_with_medians(_data, _colName, _outlier_indice):
    median_value = _data[_colName].median()
    _data.loc[_colName, _outlier_indice] = median_value
    return

def getAveragePlayerFromCluster(data, labels):
    unique_clusters = set(labels)
    unique_clusters = sorted(unique_clusters)
    centroids = [[] for x in unique_clusters]
    i = 0
    for cluster in unique_clusters:
        indices = np.where(labels == cluster)
        cluster_data = data.iloc[indices]
        centroids[i] = cluster_data.mean()
        i = i+1
    return unique_clusters, centroids


def plot(algorithm, selected_data, image_name, dpi=300):
    transformed_data = pd.DataFrame(algorithm.fit_transform(selected_data))

    transformed_data.to_csv(path_or_buf="TSNE-fitted data with all data (" + str(i) + ").csv", index=False)

    # Draw scatter plots
    print('Drawing scatter plots...')

    plot_kwds = {'alpha': 0.4, 's': 1, 'linewidths': 0}

    palette = sns.color_palette('deep', hdbscan_instance.labels_.max()+1)
    cluster_colors = [sns.desaturate(palette[col], np.clip(sat*2, 0.0, 1.0))
                      if col >= 0 else (0.95, 0.95, 0.95) for col, sat in
                      zip(hdbscan_instance.labels_, hdbscan_instance.probabilities_)]

    print("\nLabels:")
    print(sorted(collections.Counter(hdbscan_instance.labels_)))

    pyplot.scatter(x=transformed_data[0], y=transformed_data[1], color=cluster_colors, **plot_kwds)
    pyplot.title(str(hdbscan_instance.min_cluster_size))
    pyplot.savefig(image_name, dpi=dpi)
    pyplot.show()


def output_centroids(centroids):
    count = 0
    column_names = []
    for _cent in centroids:
        if count == 0:
            new_file = pd.DataFrame(_cent.values).T
            column_names = _cent.index
        else:
            new_file = new_file.append(pd.DataFrame(_cent.values).T)
        count = count + 1
    new_file.columns = column_names
    new_file.to_csv('HDBSCAN centroids.csv', index=False)

if __name__ == '__main__':
    pyplot.close('all')
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    pca = PCA(n_components=2)
    tSne = TSNE(n_components=2, init='pca', n_iter=1500, n_iter_without_progress=300, verbose=4)

    print('Loading data...')
    print("Loading file 1...")
    orig_data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)

    for i in range(2, 3):
        print("Loading file " + str(i) + "...")
        orig_data = orig_data.append(pd.read_csv('output_data/summary_data_' + str(i) + '000.csv', error_bad_lines=False))

    orig_data.reset_index(inplace=True)

    # Clean Data
    print("Removing game_size column...")
    orig_data = orig_data.drop(columns=['game_size'], axis=1)

    print("Removing players with no kills...")
    orig_data = orig_data.query('kill_count != 0')
    #orig_data = orig_data.query('survive_time > 600')
    #orig_data = orig_data.query('party_size == 2')

    # Replaced NaN values with median of the actual values
    print("Replacing NaN values in killed_from with median of the actual values...")
    median = orig_data.query('killed_from != "Nan"')['killed_from'].median()
    orig_data['killed_from'] = orig_data['killed_from'].fillna(median)

    print("Removing outliers...")
    # REMOVE OUTLIERS distance_walked
    outlier_indices = find_outliers(orig_data['distance_walked'], 5)
    #print("distance_walked outliers: ", orig_data.iloc[outlier_indices]['distance_walked'])
    orig_data.drop(orig_data.index[outlier_indices], inplace=True)

    # REMOVE OUTLIERS kill_distance
    outlier_indices = find_outliers(orig_data['kill_distance'], 5)
    #print("kill_distance outliers: ", orig_data.iloc[outlier_indices]['kill_distance'])
    orig_data.drop(orig_data.index[outlier_indices], inplace=True)

    # REMOVE OUTLIERS FROM killed_from
    outlier_indices = find_outliers(orig_data['killed_from'], 5)
    #print("killed_from outliers: ", orig_data.iloc[outlier_indices]['killed_from'])
    orig_data.drop(orig_data.index[outlier_indices], inplace=True)

    # REMOVE OUTLIERS player_dmg
    outlier_indices = find_outliers(orig_data['player_dmg'], 5)
    #print("player_dmg outliers: ", orig_data.iloc[outlier_indices]['player_dmg'])
    orig_data.drop(orig_data.index[outlier_indices], inplace=True)

    # REMOVE OUTLIERS FROM survive_time
    outlier_indices = find_outliers(orig_data['survive_time'], 5)
    #print("Survive time outliers: ", orig_data.iloc[outlier_indices]['survive_time'])
    orig_data.drop(orig_data.index[outlier_indices], inplace=True)

    orig_data.reset_index(inplace=True)

    print("Number of rows after cleaning: " + str(orig_data.__len__()))

    print("Maximum kill distance: " + str(orig_data['kill_distance'].max()))

    data = orig_data.copy(True)

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
    #data['party_size'] = normalise_column(data['party_size'], 0.0)

    print("Weighing column data...")
    #data['kill_count'] = multiply_column(data['kill_count'], 0.4)
    #data['survive_time'] = multiply_column(data['survive_time'], 0.25)
    #data['kill_knockdown_ratio'] = multiply_column(data['Zone'], 0.25)
    data['Sniper Rifle'] = multiply_column(data['Sniper Rifle'], 0.4)
    data['Carbine'] = multiply_column(data['Carbine'], 0.4)
    data['Assault Rifle'] = multiply_column(data['Assault Rifle'], 0.4)
    data['LMG'] = multiply_column(data['LMG'], 0.4)
    data['SMG'] = multiply_column(data['SMG'], 0.4)
    data['Shotgun'] = multiply_column(data['Shotgun'], 0.4)
    data['Pistols and Sidearm'] = multiply_column(data['Pistols and Sidearm'], 0.4)
    data['Melee'] = multiply_column(data['Melee'], 0.4)
    data['Crossbow'] = multiply_column(data['Crossbow'], 0.4)
    data['Throwable'] = multiply_column(data['Throwable'], 0.4)
    data['Vehicle'] = multiply_column(data['Vehicle'], 0.4)
    data['Environment'] = multiply_column(data['Environment'], 0.4)
    data['Zone'] = multiply_column(data['Zone'], 0.4)
    data['Other'] = multiply_column(data['Other'], 0.4)
    data['down and out'] = multiply_column(data['down and out'], 0.4)

    print("Selecting data columns...")
    selected_data = data[[
          'distance_walked', 'distance_rode'
        , 'travel_ratio'
        , 'kill_count'
        , 'knockdown_count', 'player_assists'
        , 'kill_knockdown_ratio'
        , 'kill_distance'
        , 'killed_from'
        , 'survive_time'
        , 'player_dmg'
        ,'team_placement'
        ,'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm', 'Melee', 'Crossbow'
        ,'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
    ]]

    selected_data.to_csv(path_or_buf="Selected data.csv", index=False)

    significance = 0.01
    print("Fitting and transforming data...")
    # transformed_data = pd.DataFrame(tSne.fit_transform(selected_data))
    # transformed_data.to_csv(path_or_buf="TSNE-fitted data with all data.csv", index=False)
    # transformed_data.to_csv(path_or_buf="TSNE-fitted data without weapons and party_size.csv", index=False)
    # transformed_data = pd.read_csv(filepath_or_buffer="TSNE-fitted data with all data.csv")
    # transformed_data = pd.read_csv(filepath_or_buffer="TSNE-fitted data without weapons and party_size.csv")

    for i in range(3, 4):
        print("Running HDBSCAN...")
        hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=int(data.__len__()*significance), min_samples=None, alpha=1.0, core_dist_n_jobs=2)
        hdbscan_instance.fit(selected_data)
        # Save HDBSCAN labels to CSV
        pd.DataFrame(hdbscan_instance.labels_).to_csv(path_or_buf="TSNE-fitted data with all data - HDBSCAN labels (" + str(i) + ").csv", index=True)
        print("# of HDBSCAN labels: " + str(hdbscan_instance.labels_.max()+1))

        print("HDBSCAN done!")

        print("\nCluster label counts:")
        cluster_label_counts = collections.Counter(hdbscan_instance.labels_)
        print("# of unclustered data points: " + str(cluster_label_counts[-1]))

        print("PCA plotting...")
        # PCA PLOTTING
        plot(pca, selected_data, 'PCA scatterplot (' + str(i) + ') ' + str(hdbscan_instance.min_samples) + ' min samples.png')

        print("Plotting condensed tree...")
        hdbscan_instance.condensed_tree_.plot()
        pyplot.savefig('Condensed tree (' + str(i) + ') ' + str(hdbscan_instance.min_samples) + ' min samples.png', dpi=300)
        pyplot.show()

        print("Getting average players...")
        unique_clusters, average_players = getAveragePlayerFromCluster(orig_data, hdbscan_instance.labels_)
        j = 0
        for x in average_players:
            print('\nCluster: ', unique_clusters[j])
            print('Cluster player medians:')
            print(x)
            j = j + 1

        output_centroids(average_players)

        # T-SNE PLOTTING
        print("T-SNE plotting...")
        plot(tSne, selected_data, 'T-SNE scatterplot (' + str(i) + ') ' + str(hdbscan_instance.min_samples) + ' min samples.png')
        #transformed_data = pd.read_csv(filepath_or_buffer="TSNE-fitted data with all data (7).csv")
        #transformed_data.to_csv(path_or_buf="TSNE-fitted data with all data (" + str(i) + ") craycray.csv", index=False)


# label_data = pd.read_csv(filepath_or_buffer="RESULTS/HDBSCAN + T-SNE (1)/3%/TSNE-fitted data with all data - HDBSCAN labels (1).csv")['label'].values
# data = pd.read_csv(filepath_or_buffer="RESULTS/HDBSCAN + T-SNE (1)/3%/Selected data.csv")
# average_players = getAveragePlayerFromCluster(data, label_data)
#
# for x in average_players:
#     print('\nCluster player medians:')
#     print(x)
