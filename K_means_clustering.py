import numpy as np
from sklearn.decomposition import PCA
import copy
import random
import matplotlib.pyplot as plt
import itertools as itertool
from collections import Counter
import sys
import math


all_weights = {'Base Damage': 1, 'Fire Rate': 1, 'DPS': 1, 'Time to Kill': 1,
               'Shots to Kill (Chest)': 1, 'Shots to Kill (Head)': 1, 'Weapon Type': 1,}

column_names = []


def SetColumnNames(column_names_list):
    global column_names
    column_names = column_names_list

#Setter for weights
def SetWeight(column_name, weight):
    for key, value in all_weights.items():
        if column_name in key:
            all_weights[key] = weight
            return
    print("WE SHOULD NOT SEE THUS**************__SET_WEIGHT")

#Getter for the all_weights dictionary, getting the weight for a string input
def GetWeightForColumnName(_col_name):
    for key, value in all_weights.items():
        if key in _col_name:
            return value
    print("WE SHOULD NOT SEE THUS**************__GET_WEIGHT")
    print(_col_name)

#calculate distance between weapons
def CalculateDistance(person1, person2, ignore_gender):
    dist = 0
    for i in range(len(person1)):
        nominal = False
        if i not in [0]:
            weight = GetWeightForColumnName(column_names[i])
            if i in [7]:
                nominal = True
            if nominal:
                if not person1[i] == person2[i]:
                    dist += 1 * weight
            else:
                dist += pow((float(person1[i]) - float(person2[i])), 2) * weight
    return math.sqrt(dist)

#Making distance matrix between all data and all data
def MakeDistanceMatrix(allData):
    distanceMatrix = np.zeros(shape=(allData.__len__(), allData.__len__()))
    for i in range(0, allData.__len__()):
        for j in range(i, allData.__len__()):
            if i != j:
                currDist = CalculateDistance(allData[i], allData[j], False)
                distanceMatrix[i][j] = currDist
                distanceMatrix[j][i] = currDist
    return distanceMatrix

def GetKey(item):
    if len(item) > 0:
        return item[0]
    else:
        return 0

#Building clusters based on the data and the centroids
def BuildClusters(allData, centroids):
    clusters = [[] for _ in centroids]
    for i in range(0, allData.__len__()):
        minDistance = float("inf")
        minIndex = -1
        for j in range(0, centroids.__len__()):
            dist = CalculateDistance(centroids[j], allData[i], False)
            dist = dist * dist
            if dist < minDistance:
                minIndex = j
                minDistance = dist
        clusters[minIndex].append(i)
    return clusters

#Check if clusters from 2 iterations are the same
def AreClustersTheSame(cluster1, cluster2):
    for old_cluster in cluster1:
        if old_cluster not in cluster2:
            return False
    return True

#Making new centroids based on all data and the new clusters
def MakeNewCentroids(allData, clusters):
    i = 0
    newCentroids = []
    for cluster in clusters:
        new_centroid = []
        for i in range(0, len(allData[0])):
            no_data = False;
            current_data = allData[cluster, i]
            if len(current_data) > 0:
                is_discrete = False
                new_val = 0
                if i in [0, 7]:
                    is_discrete = True
                if is_discrete:
                    new_val = Counter(current_data).most_common(1)[0][0]
                else:
                    new_val = np.mean(current_data.astype(np.float))
                new_centroid.append(new_val)
            else:
                no_data = True;
        # MAKE NEW VIRTUAL CENTROID MUSHROOM BASED ON MOST COMMON VALUE
        # ATTACH NEW MUSHROOM TO LIST
        if no_data:
            print("No Data")
            newCentroids.append(MakeSingleRandomCentroid(allData))
        else:
            newCentroids.append(new_centroid)
        i=i+1
    return newCentroids

#Make single random centroid
def MakeSingleRandomCentroid(allData):
    random_centroid = []
    for j in range(len(allData[0])):
        current_column = allData[:, j]
        is_discrete = True
        if j in [0, 7]:
            is_discrete = False
        if is_discrete:
            discrete = set(current_column)
            discrete = list(discrete)
            random_val = discrete[random.randint(0, len(discrete) - 1)]
        else:
            random_val = myRandom(0, 1)
        random_centroid.append(random_val)
    return random_centroid

#make all random centroids
def MakeRandomCentroids(allData, k):
    all_random_centroids = []
    for i in range(k):
        random_centroid = MakeSingleRandomCentroid(allData)
        all_random_centroids.append(random_centroid)
    return all_random_centroids

#Alternative to random centroids
def ChooseCentroidsFromDataSet(allData, k):
    centroids_from_dataset = []
    indices = []
    while len(indices) < k:
        indices.append(np.random.random_integers(0, len(allData) - 1))
        indices = list(set(indices))
    for i in indices:
        centroids_from_dataset.append(allData[i,:])
    return centroids_from_dataset

#Drawing the scatter plot based on clusters
def DrawScatterPlots(plotData, clusters):
    colors = itertool.cycle(["r", "b", "g", "c", "m", "y", "k", "w"])
    for clus in clusters:
        c = next(colors)
        for index in clus:
            plt.scatter(plotData[index, 0], plotData[index, 1], color=c, s=5)
    plt.show()

#Custom random function which includes b
def myRandom(a, b):
    candidate = random.uniform(a, b + sys.float_info.epsilon)
    while candidate > b:
       candidate = random.uniform(a, b + sys.float_info.epsilon)
    return candidate

#main fuction
def KMeansClustering(allData, clusterCount, drawPlots):
    # distanceMatrix = MakeDistanceMatrix(allData)
    # print(distanceMatrix)
    plt.close('all')
    if drawPlots:

        distanceMatrix = MakeDistanceMatrix(allData)
        pca = PCA(n_components=2)
        plotData = pca.fit_transform(distanceMatrix)
    # BUILD INITIAL CENTROIDS (RANDOM)
    random_centroids = MakeRandomCentroids(allData, clusterCount)
    #random_centroids = ChooseCentroidsFromDataSet(allData, clusterCount)
    """
    for _rand in random_centroids:
        print('rand centroids: ', _rand)
    """
    # BUILD INITIAL CLUSTERS
    clusters = BuildClusters(allData, random_centroids)
    """
    for clust in clusters:
        print("initial cluster", clust)
    """
    # PLOT INITIAL CLUSTERS
    if drawPlots:
        DrawScatterPlots(plotData, clusters)
    newClusters = []
    centroids = []
    i = 0
    # REMAKE CENTROIDS AND CLUSTERS UNTIL 2 IN A ROW ARE THE SAME
    while True:
        if newClusters:
            clusters = copy.deepcopy(newClusters)
        centroids = MakeNewCentroids(allData, clusters)
        newClusters = BuildClusters(allData, centroids)
        """
        for cent in centroids:
            print("NEW centroid: ", cent)
        for clust in newClusters:
            print("NEW cluster: ", clust)
        """
        # PLOT NEW CLUSTERS
        if drawPlots:
            DrawScatterPlots(plotData, newClusters)
        # PRINT NEW CLUSTERS
        """
        print("New K-means Iteration")
        """
        i = i+1

        if i > 10:
            print("***Clusterning could not converge***")
            break
        if AreClustersTheSame(clusters, newClusters) :
            break
    return newClusters, centroids