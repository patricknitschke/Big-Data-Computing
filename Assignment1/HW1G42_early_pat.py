from pyspark import SparkContext, SparkConf
import sys
import os

def map_phase(document):
	pairs_dict = {}
	for line in document.split('\n'):
		product, user, rate, _ = line.split(',')
		pairs_dict.update({user : (product,(rate,1))})
	return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def gather_pairs(pairs_in_list):
	pairs_dict = {}
	for pair in pairs_in_list:
		if pair[0] not in pairs_dict:
			pairs_dict[pair[0]] = [pair[1]]
		else:
			pairs_dict[pair[0]].append(pair[1])

	print("\nGathered:")		
	print(pairs_dict)
	return pairs_dict

def sum_user_ratings(user_ratings_dict):
	reduced_user_ratings = []
	for user, ratings_list in user_ratings_dict.items():
		total_rating = 0
		num_reviews = 0
		for review in ratings_list:
			#print(review)
			rating = review[1] # (rating, count)
			
			total_rating += int(rating[0])
			num_reviews += int(rating[1])

		#print(total_rating, num_reviews)

		for review in ratings_list:
			product = review[0]
			product_rating = review[1]
			reduced_user_ratings.append((user, (product, (total_rating, num_reviews)), product_rating))

		print(user, ": ", ratings_list)
	
	print("\nReduced: ")
	for reduced_item in reduced_user_ratings:
		print(reduced_item)

	return reduced_user_ratings
		

def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python G42HW1.py <K> <T> <file_name>"

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
   
    # ROUND 1 
	normalizedRatings = RawData.flatMap(map_phase)
	normalizedRatings_sample = normalizedRatings.take(7)
	print(normalizedRatings_sample)
	print()
	user_ratings = gather_pairs(normalizedRatings_sample)
	sum_user_ratings(user_ratings)

    
if __name__ == "__main__":
	main()
