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

# Similar to previous code we need to group by and count the number of cars per color and sort them in descending order
ans_4 = df.groupBy('Vehicle Color').count().orderBy(desc("count"))
ans_4.show(10)