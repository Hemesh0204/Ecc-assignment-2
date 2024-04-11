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




# Remember that we have filled all missing values with zeroes, even the column had some missing values so, I will be filtering the rows without 0 in year columns
df_2 = df.filter(df['Vehicle Year'] != 0)
df_2.count()

# Now we will group by year and vehicle make and sort them in descending order
ans_2 = df_2.groupBy('Vehicle Year', 'Vehicle Body Type').count().\
        orderBy(desc("count"))
ans_2.show(10)