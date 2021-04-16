from pyspark import SparkContext, SparkConf
import sys
import os

def map_phase(document):
	pairs_dict = {}
	for line in document.split('\n'):
		product, user, rating, timestamp = line.split(',')
		pairs_dict.update({user:(product, rating)})
	return[(key, pairs_dict[key]) for key in pairs_dict.keys()]


def normalize_rating(pairs):
	totalrating = 0
	count = 0
	product_list = []
	for p in pairs[1]:
		product, rating = p[0], int(float(p[1]))
		totalrating += rating
		count += 1
		product_list.append([product, rating])
	avg_userrating = totalrating/count
	return([(element[0], element[1]-avg_userrating) for element in product_list])    
    
def main():
    
	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python HW1G42.py <K> <T> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('G42HW1').setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	K = sys.argv[1]
	K = int(K)

    # 2. Read number of products to show
	T = sys.argv[2]
	T = int(T)
   
  
	# 3. Read input file and subdivide it into K random partitions
	data_path = sys.argv[3]
	assert os.path.isfile(data_path), "File or folder not found"
	RawData = sc.textFile(data_path,minPartitions=K).cache()
	RawData.repartition(numPartitions=K)
	print('\n INPUT PARAMETERS:  K =', K, ' T =', T, ' file =', data_path, '\n')
    
    # MAPREDUCE 
	normalizedRatings = (RawData.flatMap(map_phase)              # <-- MAP PHASE(R1)
						.groupByKey()
						.flatMap(normalize_rating))               # <-- REDUCE PHASE(R1)
	maxNormRatings = (normalizedRatings.reduceByKey(lambda x, y: max(x, y))     # <-- REDUCE PHASE(R2)
                        .sortBy(lambda a:a[1], ascending = False).take(T))
	print('OUTPUT: ')
	for x in range(len(maxNormRatings)):
		print('Product: ', maxNormRatings[x][0], '  maxNormRating: ', maxNormRatings[x][1])
    
if __name__ == "__main__":
	main()
