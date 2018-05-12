from scipy.spatial import distance_matrix
import numpy as np
import DistanceMeasurements as dm


def silhouette_coefficient(avg_dist_to_own, avg_dist_to_others):
    if avg_dist_to_own < avg_dist_to_others:
        return 1 - avg_dist_to_own / avg_dist_to_others
    elif avg_dist_to_own == avg_dist_to_others:
        return 0
    else:
        return avg_dist_to_others / avg_dist_to_own - 1


def get_average_distances(data):
    return np.average(distance_matrix(data))


