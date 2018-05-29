import numpy as np
import matplotlib.pyplot as plt

#Question 0
#load in dataset
def loadData(fileName):
    #we don't want the labels
    irisData = np.loadtxt(fileName, delimiter=',', usecols=(0, 1, 2, 3))

    return irisData

#Question 1
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

#Question 1
#find the sum of square errors
#perform the formula on each data point for each feature
def sum_of_square_error(dataSet, clusters, k, init_centroids):
    all_error_total = 0.0
    sum_of_square_error_list = []
    for i in range(0, k):
        temp = []
        for j in range(0, dataSet.shape[0]):
            if (clusters[j] == i):
                temp.append(dataSet[j])
        cluster_group = np.array(temp)
    
        number = 0.0
        
        total = 0.0
        for p in range(0, cluster_group.shape[0]):
            for k in range(0, cluster_group.shape[1]):
                number = (cluster_group[p][k] - init_centroids[i][k]) ** 2
                total = total + number
        sum_of_square_error_list.append(total)
                
        all_error_total = all_error_total + total

    sum_of_square_error_array = np.array(sum_of_square_error_list)
    print sum_of_square_error_array
    standDeviation = np.std(sum_of_square_error_array)
    print "Combined total: " + str(all_error_total)
    print "Standard Deviation: " + str(standDeviation) + "\n" 
    #print all_error_total
    return (all_error_total, standDeviation)
      


#Question 1
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

#Question 1
#to choose a set of new centroids by taking the mean of the cluster set
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

#Question 1   
# k means algorithm
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

def knee_plot():
    print "For k = 1 to 10\n"
    k_clusters_array = [1,2,3,4,5,6,7,8,9,10]
    SSE_array = []
    SD_array = []

    for i in k_clusters_array:
        knee_counter = 1
        knee_current_centroids = chooseCentroids(irisData, i)
        knee_init_centroids = knee_current_centroids

        (knee_cluster_assignment, knee_updating_centroids) = k_means_clustering(irisData, i, knee_init_centroids)
        knee_current_centroids = knee_updating_centroids

        #print ("Sum of square error: ")
        # sum of square error after the algorithm runs once
        #sum_of_square_error(irisData, cluster_assignment, i, updating_centroids)

        #runs the k means algorithm until the centroid doesn't change anymore
        while (1):
            (knee_cluster_assignment, knee_updating_centroids) = k_means_clustering(irisData, i, knee_current_centroids)
            if (np.array_equal(knee_updating_centroids, knee_current_centroids)):
                knee_final_centroids = knee_updating_centroids
                (error_total, standardDev) = (sum_of_square_error(irisData, knee_cluster_assignment, i, knee_final_centroids))
                SSE_array.append(error_total)
                SD_array.append(standardDev)
                break
            knee_current_centroids = knee_updating_centroids
            #print total_error
            knee_counter = knee_counter + 1
        
        print "Number of iterations: " + str(knee_counter) + "\n"

    plt.title("Knee Plot: SSE x Number of clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("k Clusters value")
    plt.errorbar(k_clusters_array, SSE_array, yerr=SD_array, fmt='-o')
    plt.show()
        

# here is the main
irisData = loadData('iris-data.txt')

num_clusters = input("How many k-clusters: ")

iterationCounter = 1

current_centroids = chooseCentroids(irisData, num_clusters)
init_centroids = current_centroids

(cluster_assignment, updating_centroids) = k_means_clustering(irisData, num_clusters, init_centroids)
current_centroids = updating_centroids

print ("Sum of square error: ")
# sum of square error after the algorithm runs once
#sum_of_square_error(irisData, cluster_assignment, num_clusters, updating_centroids)

#runs the k means algorithm until the centroid doesn't change anymore
while (1):
    (cluster_assignment, updating_centroids) = k_means_clustering(irisData, num_clusters, current_centroids)
    if (np.array_equal(updating_centroids, current_centroids)):
        final_centroids = updating_centroids
        sum_of_square_error(irisData, cluster_assignment, num_clusters, final_centroids)
        break
    current_centroids = updating_centroids
    #print total_error
    iterationCounter = iterationCounter + 1

# sum of square error for the final centroids
#sum_of_square_error(irisData, cluster_assignment, num_clusters, final_centroids)
print "Number of iterations: " + str(iterationCounter) + "\n"

#--------------------------------------------------------------------------------------------------------------------------

knee_plot()