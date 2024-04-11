import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, count, desc, avg, concat
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as f
from pyspark.sql.window import Window



spark = SparkSession.builder.appName('FirstDataset').getOrCreate()

df = spark.read.csv("/input", header=True, inferSchema=True)

# Lets check for the number of missing values for each columm

# null_counts = df.select([
#     count(when(isnull(c), c)).alias(c) for c in df.columns
# ]).toPandas()
# null_counts


# Since we have both numerical and categorical missing values lets find the those colums and handel the fill the missing values seperately
num_cols = [col for col, datatype in df.dtypes if datatype in ('int')]
cat_cols = [col for col in df.columns if col not in num_cols]


for feature in num_cols:
  df = df.withColumn(feature, when(col(feature).isNull(), 0).otherwise(col(feature)))

for feature in cat_cols:
  df = df.withColumn(feature, when(col(feature).isNull(), 'na').otherwise(col(feature)))

# We will be grouping the data based on the violation time and counting and displaying them in descending
df_5 = df.select('Street Code1', 'Street Code2', 'Street Code3', 'Vehicle Color')

# Combine the three feature into one single vector
assembler = VectorAssembler(inputCols=['Street Code1', 'Street Code2', 'Street Code3'], outputCol='features')
feature_vector = assembler.transform(df_5)

kmeans = KMeans(k = 3, seed = 42)
input_features = feature_vector.select('features')
model_1 = kmeans.fit(input_features)

feature_data = model_1.transform(feature_vector)
# feature_data.show()

black_codes = [
    "BLK",
    "BLACK",
    "BLAC",
    "BLCK",
    "B LK",
    "BLK.",
    "BK",
    "BK.",
    "BKG",
    "BKBL",
    "BKGY",
    "BKGR",
    "BKL",
    "BKRD",
    "BKWH",
    "BKBK",
    "BKLKK",
    "BK333", # Unsure whether it is black or not
    "BK000", # Unsure whether it is black or not
    "BK/TN", # Unsure whether it is black or not
    "BK/RD", # Unsure whether it is black or not
    "BK/GR", # Unsure whether it is black or not
    "BK/WH",
    "BKGL", # Unsure whether it is black or not
    "BKGRA", # Unsure whether it is black or not
    "BLK/U", # Unsure whether it is black or not
    "BLKOT", # Unsure whether it is black or not
    "BLC",
    "B LAC", # Unsure whether it is black or not
    "BLKG",
    "BLKRE",
    "BLKGR", # Unsure whether it is black or not
    "BLKR", # Unsure whether it is black or not
    "BLKA",
    "BLKD",
    "BBLACK", # Unsure whether it is black or not
    "BLKDK",
]
# now lets count for each cluster how many black vehicles vs total count of vehicle
black_cars_df = feature_data.groupBy('prediction').\
    agg(
        count(when(col('Vehicle Color').isin(black_codes), 1)).alias("Number of black cars in the cluster"),
        count('Vehicle Color').alias("Total number of cars for cluster")
    ).orderBy("prediction")
# black_cars_df.show()

target_codes = [34510.0, 10030.0, 34050.0]
start_index = -1
distance = np.inf
models_centroid = np.array(model_1.clusterCenters()).astype(float)
for index, clusters_center in enumerate(models_centroid):
    euclidian_distance = np.sum(np.square(np.subtract(target_codes, clusters_center)))
    if euclidian_distance < distance:
        distance = euclidian_distance
        ans = index


print(f"Closet Centroid for given data: {ans}")