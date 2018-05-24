import numpy as np
import matplotlib.pyplot as plt

def loadData(fileName):
    #we don't want the labels
    irisData = np.loadtxt(fileName, delimiter=',', usecols=(0, 1, 2, 3))

    return irisData

# Lp norm distance formula
# From assignment 1
# p = 2 in this assignment because we want Euclidean distance
def distance(x,y,p):
    array1 = x
    array2 = y
    temp = 0
    total = 0
    distance = 0

    for i in range(len(array1)):
        temp = abs(array1[i] - array2[i]) ** p
        total = total + temp
    
    distance = total ** (float(1)/p)
    return distance

def sum_of_square_error(dataSet, clusters, k, init_centroids):
    for i in range(0, k):
        temp = []
        for j in range(0, dataSet.shape[0]):
            if (clusters[j] == i):
                temp.append(dataSet[j])
        cluster_group = np.array(temp)
    
        number = 0.0
        sum_of_square_error_list = []
        total = 0.0
        for p in range(0, cluster_group.shape[0]):
            for k in range(0, cluster_group.shape[1]):
                number = (cluster_group[p][k] - init_centroids[i][k]) ** 2
                total = total + number
        sum_of_square_error_list.append(total)
        sum_of_square_error_array = np.array(sum_of_square_error_list)
        #print total
        print sum_of_square_error_array
    #return total          


#k is the number of clusters
def chooseCentroids(dataSet, k):
    init_centroids_List = []

    #i don't want to shuffle the whole dataSet because i might want to keep track of the indexes later
    #so i shuffle the index instead
    shuffle_index = np.arange(dataSet.shape[0])
    np.random.shuffle(shuffle_index)

    for i in range(0, k):
        init_centroids_List.append(dataSet[shuffle_index[i]])

    init_centroids = np.array(init_centroids_List)
    # print init_centroids
    return init_centroids

def reevaluateCenters(dataSet, clusters, k):
    all_new_centers_list = []
    for i in range(0, k):
        temp = []
        for j in range(0, dataSet.shape[0]):
            if (clusters[j] == i):
                temp.append(dataSet[j])
        cluster_group = np.array(temp)

        #sum_of_square_error(cluster_group, init_centroids)

        new_center_list = []
        for p in range(0, dataSet.shape[1]):
            mean_of_cols = np.mean(cluster_group[:,p])
            new_center_list.append(mean_of_cols) 
        #new_centers = np.array(new_center_list)
        all_new_centers_list.append(new_center_list)
    
    all_new_center = np.array(all_new_centers_list)
    #print all_new_center
    return all_new_center
    

#x_input is the dataset
#k is the number of clusters
#init_centroids
def k_means_clustering(x_input, k, init_centroids):
    #previous_centroids_array = init_centroids
    cluster_assignments_array = []
    
    for i in range (0, x_input.shape[0]):
        tempArray = x_input[i]
        distanceTemp = 1000000
        shortestDist = 1000000
        indexTemp = -1
        for j in range(0, init_centroids.shape[0]):
            # use p = 2 because we want Euclidean distance
            distanceTemp = distance(tempArray, init_centroids[j], 2)
            
            if (distanceTemp < shortestDist):
                # print j
                shortestDist = distanceTemp
                indexTemp = j
        
        cluster_assignments_array.append(indexTemp)
    cluster_assignments = np.array(cluster_assignments_array)
    
    
    updating_centroids_array = reevaluateCenters(x_input, cluster_assignments, k)
    return (cluster_assignments, updating_centroids_array)

        

# here is the main
irisData = loadData('iris-data.txt')

num_clusters = input("How many k-clusters: ")

current_centroids = chooseCentroids(irisData, num_clusters)
init_centroids = current_centroids

(cluster_assignment, updating_centroids) = k_means_clustering(irisData, num_clusters, init_centroids)
#this is bad implementation
current_centroids = updating_centroids
print ("Sum of square error: ")
sum_of_square_error(irisData, cluster_assignment, num_clusters, updating_centroids)
print ("\n")

while (1):
    (cluster_assignment, updating_centroids) = k_means_clustering(irisData, num_clusters, current_centroids)
    if (np.array_equal(updating_centroids, current_centroids)):
        final_centroids = updating_centroids
        break
    current_centroids = updating_centroids

sum_of_square_error(irisData, cluster_assignment, num_clusters, final_centroids)
