from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, count, avg, concat
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


# Fill the zone information based on the shot distance, close def distance, low clock
df_7 = nba_data.withColumn(
    'zone',
    concat(
        f.when(nba_data.SHOT_DIST <= 15, '_Close_range_shot').otherwise('_Long_shot'),
        f.when(nba_data.SHOT_CLOCK <= 12, '_Quick').otherwise('_Slow'),
        f.when(nba_data.CLOSE_DEF_DIST <= 5, '_close_by').otherwise('_distant')  
    )
)

# Calculate ratio for each player in different zones
ratio_for_player_for_zones = df_7.filter(
    df_7.player_name.isin(['james harden', 'chris paul', 'stephen curry', 'lebron james'])
).groupBy('player_name', 'zone').agg(
    f.sum('FGM'),
    f.count('SHOT_NUMBER').alias('Total count of shots'),
    (f.sum('FGM') / f.count('SHOT_NUMBER')).alias('hit_rate')
)
# ratio_for_player_for_zones.show()

# Get the best zone for each player based on the ratio of above table
player_best_zones_df = ratio_for_player_for_zones.withColumn(
    'rank',
    f.rank().over(Window.partitionBy('player_name').orderBy(f.desc('hit_rate')))
).filter(f.col('rank') == 1).drop('rank')
player_best_zones_df.show()