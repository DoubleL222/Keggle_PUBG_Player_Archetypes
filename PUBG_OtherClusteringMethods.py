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

pyplot.close('all')

K = 10

data = pd.read_csv('output_data/summary_data_1000.csv', error_bad_lines=False)
for i in range(2,150):
    data.append(pd.read_csv('output_data/summary_data_'+ str(i)+ '000.csv', error_bad_lines=False))

data = data.drop(columns=['game_size'], axis=1)

median = data.query('killed_from != "Nan"')['killed_from'].median()
data['killed_from'] = data['killed_from'].fillna(median)

# REMOVE OUTLIERS DISTANCE WALKED
outlier_indices = PUBG_DataPlotter.find_outliers(data['distance_walked'], 5)
print("distance_walked outliers: ", data.iloc[outlier_indices]['distance_walked'])
data.drop(data.index[outlier_indices], inplace=True)

# REMOVE OUTLIERS DISTANCE RODE
# outlier_indices = PUBG_DataPlotter.find_outliers(data['distance_rode'], 5)
# print("distance_rode outliers: ", data.iloc[outlier_indices]['distance_rode'])
# data.drop(data.index[outlier_indices], inplace=True)

# REMOVE OUTLIERS KILL DISTANCE RODE
outlier_indices = PUBG_DataPlotter.find_outliers(data['kill_distance'], 8)
print("kill_distance outliers: ", data.iloc[outlier_indices]['kill_distance'])
data.drop(data.index[outlier_indices], inplace=True)

# REMOVE OUTLIERS KILL FROM RODE
outlier_indices = PUBG_DataPlotter.find_outliers(data['killed_from'], 5)
print("killed_from outliers: ", data.iloc[outlier_indices]['killed_from'])
data.drop(data.index[outlier_indices], inplace=True)

# REMOVE OUTLIERS KILL FROM RODE
outlier_indices = PUBG_DataPlotter.find_outliers(data['player_dmg'], 5)
print("player_dmg outliers: ", data.iloc[outlier_indices]['player_dmg'])
data.drop(data.index[outlier_indices], inplace=True)

# REMOVE OUTLIERS KILL FROM RODE
outlier_indices = PUBG_DataPlotter.find_outliers(data['survive_time'], 5)
print("Survive time outliers: ", data.iloc[outlier_indices]['survive_time'])
data.drop(data.index[outlier_indices], inplace=True)
data.reset_index(inplace=True)

#COPY
selected_data = data.__deepcopy__()

selected_data = PUBG_DataPlotter.clean_the_data(selected_data, [
    'distance_walked', 'distance_rode',
    'travel_ratio', 'kill_count',
    'knockdown_count', 'player_assists',
    'kill_knockdown_ratio', 'kill_distance',
    'survive_time', 'player_dmg', 'killed_from', 'team_placement'
    , 'party_size'
    , 'Sniper Rifle', 'Carbine', 'Assault Rifle', 'LMG', 'SMG', 'Shotgun', 'Pistols and Sidearm', 'Melee', 'Crossbow'
    , 'Throwable', 'Vehicle', 'Environment', 'Zone', 'Other', 'down and out'
])

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
    unique_clusters, centroids = getAveragePlayerFromCluster(data, labels)

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


clf = tree.DecisionTreeClassifier()
clf.fit(numpy_selected_data[:int(numpy_selected_data.shape[0]/2), :], labels[:int(numpy_selected_data.shape[0]/2)])
prediction = clf.predict(numpy_selected_data[int(numpy_selected_data.shape[0]/2):, :])
print("Accuracy: ", accuracy_score(labels[int(numpy_selected_data.shape[0]/2):], prediction))
print(column_names)
unique_clusters = list(unique_clusters)
string_clusters = []
for class_index in unique_clusters:
    string_clusters.append(str(class_index))
column_names = list(column_names)
column_names.__delitem__(0)
#pred_tree = tree.export_graphviz(clf, out_file=None, feature_names=column_names, class_names=string_clusters, filled=True, rounded=True)
pred_tree = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(pred_tree)
graph.render('prediction')
graph