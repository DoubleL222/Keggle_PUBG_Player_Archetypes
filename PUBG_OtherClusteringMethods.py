from sklearn.cluster import KMeans
import numpy as np
import PUBG_DataPlotter
import pandas as pd
import collections


def getAveragePlayerFromCluster(data, labels):
    unique_clusters = set(labels)
    centroids = [[] for x in unique_clusters]
    i = 0
    for cluster in unique_clusters:
        indices = np.where(labels == cluster)
        cluster_data = data.iloc[indices]
        centroids[i] = cluster_data.mean()
        i = i+1
    return centroids


data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)
selected_data = PUBG_DataPlotter.clean_the_data(data, [
    'distance_walked', 'distance_rode',
    'travel_ratio', 'kill_count',
    'knockdown_count', 'player_assists',
    'kill_knockdown_ratio', 'kill_distance',
    'survive_time', 'player_dmg', 'team_placement'
    , 'party_size'
    , 'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm', 'Melee', 'Crossbow'
    , 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
])

cluster_data = selected_data.iloc[[1,2,3,4]]
print(cluster_data)
print(cluster_data.mean())
numpy_selected_data = selected_data.as_matrix()
kmeans = KMeans(n_clusters=8).fit(numpy_selected_data)
labels = kmeans.labels_
label_counter = collections.Counter(labels)
print(label_counter)
centroids = getAveragePlayerFromCluster(selected_data, labels)
for _cent in centroids:
    print(_cent)

