from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel

import sys
import os
import random
import math
import time

'''
To read the input text file (e.g., inputPath) containing a clustering
into the RDD full clustering do:

inputPoints = sc.textFile(inputPath).map(strToTuple)
'''

def dist(a,b):
	return sum((a[i]-b[i])**2 for i in range(len(a)))

def strToTuple(line):
	return tuple(float(xi) for xi in line.strip().split())	# returns point (x1,x2)

def main():
	print("\n\n\n--------START-----\n\n\n")
	
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 7, "Usage: python G42HW3.py <file_name> <kstart> <h> <iter> <M> <L>"

	# SPARK SETUP
	#conf = (SparkConf().setAppName('Homework3').set('spark.locality.wait','0s')) for CloudVeneto
	conf = SparkConf().setAppName('G42HW3').setMaster("local[*]")
	sc = SparkContext(conf=conf)
	sc.setLogLevel("OFF")

	# INPUT READING
    
	# 1. Read input file and subdivide it into partitions
	inputPath = sys.argv[1]
	assert os.path.isfile(inputPath)
    
	# 2. Read the initial number of clusters
	kstart = sys.argv[2]
	kstart = int(kstart)

    # 3. Read the number of values of k that the program will test
	h = sys.argv[3]
	h = int(h)

	# 4. Read the number of iterations of Lloyd's algorithm.
	iterL = sys.argv[4]
	iterL = int(iterL)

	# 5. Read the expected size of the sample used to approximate the silhouette coefficient.
	M = sys.argv[5]
	M = int(M)

	# 6. Read the number of partitions of the RDDs containing the input points and their clustering.
	L = sys.argv[6]
	L = int(L)

	print(f"\nINPUT PARAMETERS:  kstart = {kstart}, h = {h}, iterL = {iterL}, M = {M}, L = {L}\n")

	# TASK 1 #

	# Store set of points into inputPoints, cached and in L partitions
	t0 = time.time()
	inputPoints = sc.textFile(inputPath).map(strToTuple).cache()
	t1 = time.time()
	
	inputPoints.repartition(numPartitions=L)
	# l = inputPoints.collect()
	# print(f"\n\n\nResult after collect: {l}\nSize: {len(l)}")
	
	print(f"Time for input reading: {round((t1-t0)*1000, 0)}\n")

	# TASK 2 #
	for k in range(kstart, kstart+h):
		
		# 2.1 Compute currentClustering with Lloyd's algorithm #

		# clusterCenters as a dictionary, index: point
		clusterCenters = sc.broadcast(KMeans.train(inputPoints, k, maxIterations=iterL, initializationMode="random").clusterCenters)
		
		def assign(p):
			# for each point
			# compute distances for each cluster center
			# return (x, min(cluster dist))
			min_cluster_index, min_dist = -1, 999999
			for i, cluster_center in enumerate(clusterCenters.value):
				center_dist = dist(p, cluster_center)
				if center_dist < min_dist:
					min_dist = center_dist
					min_cluster_index = i
			
			assert min_cluster_index != -1, "No cluster center assigned"
			return (p, min_cluster_index)

		t2 = time.time()
		currentClustering = inputPoints.map(assign)
		#print(currentClustering.take(10))
		t3 = time.time()
	

		# 2.2 Compute approximate average Silhouette Coefficient #
		t = M/k

		# SAMPLE EXTRACTION
		sharedClusterSizes = sc.broadcast(currentClustering.map(lambda tpl : tpl[1]).countByValue())
		print("SharedClusterSizes = ", sharedClusterSizes.value)
		
		def sampling(pair):
			prob = t / sharedClusterSizes.value[pair[1]]
			if(prob > 1): prob = 1
			rand = random.random()
			if (rand < prob):
				# print("\n Will take this point: ", pair)
				return pair
			else:
				# print("\n Discarded: ", pair)
				return []

		clusteringSample = sc.broadcast(currentClustering.map(sampling).filter(bool).collect())
		clusteringSampleSizes = currentClustering.map(sampling).filter(bool).map(lambda tpl : tpl[1]).countByValue()
		print("clusteringSampleSizes = ", clusteringSampleSizes)
		
		
		# SILHOUETTE COMPUTATION
		def silhCoeffOnSample(pair):
			cluster = pair[1]
			cluster_distances = [0] * k 
			for p in clusteringSample.value:
				cluster_distances[p[1]] += dist(pair[0], p[0])
			denominators = [min(t, sharedClusterSizes.value[i]) for i in range(k)]
			a_p = cluster_distances[cluster]/denominators[cluster]
			b_p = min(cluster_distances[i] / denominators[i] for i in range(k) if i != cluster)
			s_p = (b_p - a_p)/max(a_p, b_p)
			return (0, (1, s_p))

		t4 = time.time()
		approxSilhFull = currentClustering.map(silhCoeffOnSample).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).map(lambda x: x[1][1]/x[1][0]).collect()
		t5 = time.time()

		print("Number of clusters k =", k)
		print("Silhouette coefficient =", round(approxSilhFull[0],5))
		print("Time for clustering =", int(round((t3-t2)*1000, 0)))
		print("Time for silhouette computation =", int(round((t5-t4)*1000, 0)))

    
if __name__ == "__main__":
	main()