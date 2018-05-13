from sklearn.cluster import KMeans
import numpy as np
import PUBG_DataPlotter
import pandas as pd
import collections
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import tree
import graphviz
from sklearn.decomposition import PCA

def multiply_column(_col, _multiplier):
    new_col = []
    for _val in _col:
        new_col.append(_val*_multiplier)
    return np.array(new_col)

def getAveragePlayerFromCluster(data, labels):
    unique_clusters = set(labels)
    unique_clusters = sorted(unique_clusters)
    centroids = [[] for x in unique_clusters]
    i = 0
    for cluster in unique_clusters:
        print("Calculating mean for cluster: ", str(i))
        indices = np.where(labels == cluster)
        cluster_data = data.iloc[indices]
        centroids[i] = cluster_data.mean()
        i = i+1
    return unique_clusters, centroids

pyplot.close('all')

K = 10
data = pd.read_csv('all_data.csv', error_bad_lines=False)
print(len(data.index))
data = data.query('kill_count != 0')
data = data.query('party_size == 4')
print(len(data.index))

print("removing players with -1 kill_distance")
median = data.query('kill_distance != -1')['kill_distance'].median()
data['kill_distance'].replace(to_replace=-1, value=median, inplace=True)
print("kill_distance min: ", data['kill_distance'].min())

print("removing players with -1 killed_from")
median = data.query('killed_from != -1')['killed_from'].median()
data['killed_from'].replace(-1, median, inplace=True)
print("killed_from min: ", data['killed_from'].min())
'''
data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)
for i in range(2,150):
    data = data.append(pd.read_csv('output_data/summary_data_'+ str(i)+ '000.csv', error_bad_lines=False))
    print("file: ", str(i))

data = data.drop(columns=['game_size'], axis=1)

print("removing Nan from killed from")
median = data.query('killed_from != "Nan"')['killed_from'].median()
data['killed_from'] = data['killed_from'].fillna(median)

# REMOVE OUTLIERS DISTANCE RODE
# outlier_indices = PUBG_DataPlotter.find_outliers(data['distance_rode'], 5)
# print("distance_rode outliers: ", data.iloc[outlier_indices]['distance_rode'])
# data.drop(data.index[outlier_indices], inplace=True)
'''
#data.to_csv("all_data.csv", index=False)

all_outliers = set()

# REMOVE OUTLIERS distance_walked
print("Removing distance_walked outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['distance_walked'], 3)
#print(data['distance_walked'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS distance_walked
print("Removing distance_rode outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['distance_rode'], 4)
print(data['distance_rode'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS kill_distance
print("Removing kill_distance outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['kill_distance'], 10)
#print(data['kill_distance'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS FROM killed_from
print("Removing killed_from outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['killed_from'], 10)
#print(data['killed_from'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS player_dmg
print("Removing player_dmg outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['player_dmg'], 10)
#print(data['player_dmg'].iloc[new_outliers], "; kill count: ",data['kill_count'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS kill_count
print("Removing kill_count outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['kill_count'], 12)
print(data['kill_count'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS kill_count
print("Removing knockdown_count outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['knockdown_count'], 12)
print(data['knockdown_count'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# REMOVE OUTLIERS FROM survive_time
print("Removing survive_time outliers...")
new_outliers = PUBG_DataPlotter.find_outliers(data['survive_time'], 1)
#print(data['survive_time'].iloc[new_outliers])
all_outliers = all_outliers.union(new_outliers)

# print("Survive time outliers: ", data.iloc[outlier_indices]['survive_time'])
data.drop(data.index[list(all_outliers)], inplace=True)

data.reset_index(inplace=True)

#COPY
selected_data = data.__deepcopy__(True)
print("Selected data len: ",len(selected_data.index))

selected_data_columns = [
    'distance_walked', 'distance_rode',
    'travel_ratio', 'kill_count',
    'knockdown_count', 'player_assists',
    'kill_knockdown_ratio', 'kill_distance',
    'survive_time', 'player_dmg', 'killed_from', 'team_placement'
    , 'party_size'
    , 'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm', 'Melee', 'Crossbow'
    , 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
]
non_normalized_data = data[selected_data_columns]
selected_data = PUBG_DataPlotter.clean_the_data(selected_data, selected_data_columns)

selected_data['Sniper Rifle'] = multiply_column(selected_data['Sniper Rifle'], 0.2)
selected_data['Carbine'] = multiply_column(selected_data['Carbine'], 0.2)
selected_data['Assault Rifle'] = multiply_column(selected_data['Assault Rifle'], 0.2)
selected_data['LMG'] = multiply_column(selected_data['LMG'], 0.2)
selected_data['SMG'] = multiply_column(selected_data['SMG'], 0.2)
selected_data['Shotgun'] = multiply_column(selected_data['Shotgun'], 0.2)
selected_data['Pistols and Sidearm'] = multiply_column(selected_data['Pistols and Sidearm'], 0.2)
selected_data['Melee'] = multiply_column(selected_data['Melee'], 0.2)
selected_data['Crossbow'] = multiply_column(selected_data['Crossbow'], 0.2)
selected_data['Throwable'] = multiply_column(selected_data['Throwable'], 0.2)
selected_data['Vehicle'] = multiply_column(selected_data['Vehicle'], 0.2)
selected_data['Environment'] = multiply_column(selected_data['Environment'], 0.2)
selected_data['Zone'] = multiply_column(selected_data['Zone'], 0.2)
selected_data['Other'] = multiply_column(selected_data['Other'], 0.2)
selected_data['down and out'] = multiply_column(selected_data['down and out'], 0.2)

#non_normalized_data.to_csv("KMeans_non_normalized_data.csv", index=False)
#selected_data.to_csv("KMeans_normalized_data.csv", index=False)

non_normalized_numpy_data = non_normalized_data.as_matrix()
numpy_selected_data = selected_data.as_matrix()


'''
tSne = TSNE(n_components=2, init='pca', n_iter=1500, n_iter_without_progress=300, verbose=4)
plot_data = tSne.fit_transform(numpy_selected_data)

np.savetxt('set1_tsne.csv', plot_data, delimiter=',')

pca = PCA(n_components=2)
plot_data = pca.fit_transform(numpy_selected_data)
'''
column_names = []
unique_clusters = []
for K in range(8,9):
    print('\n\n\nK = ',K)
    kmeans = KMeans(n_clusters=K).fit(numpy_selected_data)
    labels = kmeans.labels_
    save_name = 'Set1_KMeans_K='+str(K)+'.csv'
    np.savetxt(save_name, labels, delimiter=',')
    label_counter = collections.Counter(labels)
    print(label_counter)
    unique_clusters, centroids = getAveragePlayerFromCluster(non_normalized_data, labels)

    print("Start calculating centroids")
    i = 0
    column_names = []
    for _cent in centroids:
        if i == 0:
            new_file = pd.DataFrame(_cent.values).T
            column_names = _cent.index
        else:
            new_file = new_file.append(pd.DataFrame(_cent.values).T)
        i = i+1
    new_file.columns = column_names
    new_file.to_csv('KMeans_K='+str(K)+'.csv', index=False)
    unique_clusters = set(labels)
    palette = sns.color_palette('hls', len(unique_clusters))
    cluster_colors = [palette[col] for col in labels]

    #plot_kwds = {'alpha': 0.4, 's': 1, 'linewidths': 0}
    #pyplot.scatter(plot_data[:, 0], plot_data[:, 1], color=cluster_colors, **plot_kwds)
    #pyplot.savefig('scatterplot_KMeans_K=' + str(K) + '.png', dpi=300)
    #pyplot.show()

print("Starting ID3")
clf = tree.DecisionTreeClassifier()
clf.fit(non_normalized_numpy_data[:int(non_normalized_numpy_data.shape[0]/2), :], labels[:int(non_normalized_numpy_data.shape[0]/2)])
prediction = clf.predict(non_normalized_numpy_data[int(non_normalized_numpy_data.shape[0]/2):, :])
print("Accuracy: ", accuracy_score(labels[int(non_normalized_numpy_data.shape[0]/2):], prediction))
print(column_names)
unique_clusters = list(unique_clusters)
string_clusters = []
for class_index in unique_clusters:
    string_clusters.append(str(class_index))
column_names = list(column_names)
#pred_tree = tree.export_graphviz(clf, out_file=None, feature_names=column_names, class_names=string_clusters, filled=True, rounded=True)
pred_tree = tree.export_graphviz(clf, out_file=None, feature_names=column_names, class_names=string_clusters, filled=True)
graph = graphviz.Source(pred_tree)
graph.render('prediction')
graph