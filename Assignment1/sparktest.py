from pyspark import SparkContext, SparkConf 

conf = SparkConf().setAppName("test").setMaster("local") 
sc = SparkContext.getOrCreate(conf = conf)


data = """B00007AVYI,A0068533X8Y5TYUJWWIC,5.0,1401148800
B001LF4EVO,A02292976AOUT3I4ZLFA,5.0,1383004800
B000PIT0RG,A0328927TA7ECTIKNP3G,4.0,1398643200
B00005N7TG,A0333047WGK24IZKLDP2,4.0,1411603200
B000IOEK7M,A0479411TZTHITJ9TFB8,5.0,1446076800
B00005N7Q5,A0534350D18UHJKPKZ1W,5.0,1437177600"""

def transform_RD(raw_data):
    pairs_dict = {}
    for line in raw_data.split('\n'):
        p, u, r, t = line.split(',')
        print(p,u,r,t)
    return pairs_dict

data_new = transform_RD(data)
print(data_new)