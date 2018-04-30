import numpy as np
from numpy import genfromtxt
import DataCleaning as dc
import K_means_clustering as Kmeans

#allData = np.genfromtxt('Data/agg_match_stats_0.csv')

my_data = genfromtxt('Data/Weapons.csv', delimiter=',')
allData = []
for line in open('Data/Weapons.csv'):
    newData = line.split(",")
    addData = []
    for entry in newData:
        addData.append(entry.strip('\'').strip())
    allData.append(addData)

allData = np.array(allData)
columnNames = allData[0][:]
print(columnNames)
allData = np.delete(allData, 0, 0)

for column_index in [1,2,3,4,5,6]:
    pick_number = allData[:, column_index]
    pick_number = dc.normalise_column(pick_number.astype(np.float))
    allData[:, column_index] = pick_number


#CLUSTERING WEAPONS
Kmeans.SetColumnNames(columnNames)
Kmeans.SetWeight('Base Damage', 1.0)
Kmeans.SetWeight('Fire Rate', 1.0)
Kmeans.SetWeight('DPS', 1.0)
Kmeans.SetWeight('Time to Kill', 1.0)
Kmeans.SetWeight('Shots to Kill (Chest)', 1.0)
Kmeans.SetWeight('Shots to Kill (Head)', 1.0)
Kmeans.SetWeight('Weapon Type', 1.0)


clusters, centroids = Kmeans.KMeansClustering(allData, 6, True)
cluster_column = np.empty((allData.shape[0],1))
full_column_names = np.append(columnNames, 'Cluster')
i=0
for cluster in clusters:
    print("-----------CLUSTER ", i,"-----------")
    print(allData[cluster,0])
    print(allData[cluster,7])
    cluster_column[cluster] = i
    i = i+1
all_data_with_columns = np.hstack((allData, cluster_column))

