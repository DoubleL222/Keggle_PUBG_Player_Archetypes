from math import sqrt

# This function takes a n-dimensional list of coordinates and returns euclidean distance of these


def euclidean_distance(coordinate_list):
    output = 0

    for dimension in coordinate_list:
        output += pow(dimension[0] - dimension[1], 2)

    return sqrt(output)


