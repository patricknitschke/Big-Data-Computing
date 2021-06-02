from pyspark import SparkContext, SparkConf
import sys
import os
import random
import math
import time

'''
To read the input text file (e.g., inputPath) containing a clustering
into the RDD full clustering do:

fullClustering = sc.textFile(inputPath).map(strToTuple)
'''


def dist(a,b):
	return sum((a[i]-b[i])**2 for i in range(len(a)))
    
def strToTuple(line):
    ch = line.strip().split(",")
    point = tuple(float(ch[i]) for i in range(len(ch)-1))
    return (point, int(ch[-1])) # returns (point, cluster_index)

def main():
	print("\n\n\n--------START-----\n\n\n")
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 6, "Usage: python G42HW3.py <file_name> <kstart> <h> <iter> <M> <L>"

	# SPARK SETUP
	#conf = (SparkConf().setAppName('Homework3').set('spark.locality.wait','0s')) for CloudVeneto
	conf = SparkConf().setAppName('G42HW3').setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# INPUT READING
    
	# 1. Read input file and subdivide it into partitions
	inputPath = sys.argv[1]
	assert os.path.isfile(inputPath)
	fullClustering = sc.textFile(inputPath).map(strToTuple).cache()
	fullClustering.repartition(numPartitions=4)
    
	# 2. Read number of clusters
	k = sys.argv[2]
	k = int(k)

    # 3. Read sample size per cluster
	t = sys.argv[3]
	t = int(t)
    
	print('\nINPUT PARAMETERS:  k =', k, ' t =', t, ' file =', inputPath, '\n')
    

    
    # SAMPLE EXTRACTION
	sharedClusterSizes = sc.broadcast(fullClustering.map(lambda tpl : tpl[1]).countByValue())
	print("\nSharedClusterSizes = ", sharedClusterSizes.value)
	def sampling(pair):
		prob = t / sharedClusterSizes.value[pair[1]]
		if(prob > 1): prob = 1
		rand = random.random()
		if (rand < prob):
			#print("\n Will take this point: ", pair)
			return pair
		else:
			#print("\n Discarded: ", pair)
			return []

	clusteringSample = sc.broadcast(fullClustering.map(sampling).filter(bool).collect())
    
	#print("\n\n\nResult after sampling: \n\n\n", clusteringSample.value)
	#print("Size:", len(clusteringSample.value), "\n\n\n")
    
    
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


	def exactCompute(clusteringSample):
		silList = []
		for pair in clusteringSample.value:
			cluster = pair[1]
			cluster_distances = [0] * k 
			for p in clusteringSample.value:
				cluster_distances[p[1]] += dist(pair[0], p[0])		
			denominators = [min(t, sharedClusterSizes.value[i]) for i in range(k)]
			a_p = cluster_distances[cluster]/denominators[cluster]
			b_p = min((cluster_distances[i] / denominators[i] for i in range(k) if i != cluster))
			s_p = (b_p - a_p)/max(a_p, b_p)
			silList.append(s_p)
			
		return sum(silList)/len(silList)
            
            
	t0 = time.time()
	approxSilhFull = fullClustering.map(silhCoeffOnSample).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).map(lambda x: x[1][1]/x[1][0]).collect()
	t1 = time.time()

	t2 = time.time()
	exactSilhSample = exactCompute(clusteringSample)   
	t3 = time.time()

	print("Values of approxSilhFull =", round(approxSilhFull[0],5))
	print("Time to compute approxSilhFull =", int(round((t1-t0)*1000, 0)), "ms")
	print("Values of exactSilhSample =", round(exactSilhSample, 5))
	print("Time to compute exactSilhSample =", int(round((t3-t2)*1000, 0)), "ms")
    
if __name__ == "__main__":
	main()