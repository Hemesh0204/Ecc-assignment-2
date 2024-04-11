from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, count, avg
from pyspark.sql.types import IntegerType, StringType
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as f

spark = SparkSession.builder.appName("NBA_data").getOrCreate()

nba_data = spark.read.csv("/FileStore/tables/shot_logs-1.csv", inferSchema = True, header = True)

numerical_cols = [cols for cols, datatype in nba_data.dtypes if datatype in ('int', 'double')]
# Lets fill the null values with average of the column since we have less missing values to handel
# Calculate the average for all the numerical valeus
averages = {col: nba_data.select(avg(col)).first()[0] for col in numerical_cols}

# Now, fill the null values with the respective averages
for feature in numerical_cols:
    nba_data = nba_data.withColumn(feature, when(col(feature).isNull(), averages[feature]).otherwise(col(feature)))


# We will get the player name and thier nearest defender and sum of the FGM and number of shots and from that calculate the ratio
players_stats = nba_data.groupBy('player_name', col('CLOSEST_DEFENDER').alias("Kuppa Paiyan")).agg(
    f.sum('FGM'),
    f.count('SHOT_NUMBER')
)


# Calculate hit rate for each defender
players_stats = players_stats.withColumn('ratio', f.col('sum(FGM)') / f.col('count(SHOT_NUMBER)').cast('double')).orderBy("ratio")
players_stats.show(10)