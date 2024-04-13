import sys
import scipy
import pandas as pd
import numpy as np

from functools import reduce
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.functions import col, when, udf, count, desc, avg, concat, sum, regexp_replace, expr, first
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("NBA_data").getOrCreate()

nba_data = spark.read.csv("/FileStore/tables/shot_logs-1.csv", inferSchema = True, header = True)

numerical_cols = [cols for cols, datatype in nba_data.dtypes if datatype in ('int', 'double')]
# Lets fill the null values with average of the column since we have less missing values to handel
# Calculate the average for all the numerical valeus
averages = {col: nba_data.select(avg(col)).first()[0] for col in numerical_cols}

# Now, fill the null values with the respective averages
for feature in numerical_cols:
    nba_data = nba_data.withColumn(feature, when(col(feature).isNull(), averages[feature]).otherwise(col(feature)))





# Combine the players by their ids and closest defender and caculate the how many times the player has made or missed the sot when an particular defender is nearby

nba_grouped_ppid = spark.sql('''SELECT player_id AS PlayerID, CLOSEST_DEFENDER_PLAYER_ID AS DefenderID,
                        SUM(CASE WHEN SHOT_RESULT = 'made' THEN 1 ELSE 0 END) AS made_shot,
                        SUM(CASE WHEN SHOT_RESULT = 'missed' THEN 1 ELSE 0 END) AS missed_shot FROM nbatemp
                        GROUP BY player_id, CLOSEST_DEFENDER_PLAYER_ID''')
# nba_grouped_ppid.show()


# Calculate the ratio for each defender and player pair and removing the duplicates, and remove the null values
nba_grouped_ppid = nba_grouped_ppid.withColumn("Ratio", expr("made_shot / (made_shot + missed_shot)"))
nba_grouped_ppid = nba_grouped_ppid.dropDuplicates(["PlayerID", "Ratio"]).filter("Ratio is not null")
# nba_grouped_ppid.show()

# We will now group each player and calculate min ratio for combo(pair of defender and player) and get the minimum
df_6 = nba_grouped_ppid.groupBy("PlayerID").agg({"Ratio": "min"}).withColumn("Ratio", col("min(Ratio)"))

combined_data = nba_grouped_ppid.join(df_6, ["PlayerID", "Ratio"]).drop("Ratio").select("PlayerID", "DefenderID")

combined_data = combined_data.join(nba_data, (combined_data["PlayerID"] == nba_data["player_id"]) & (combined_data["DefenderID"] == nba_data["CLOSEST_DEFENDER_PLAYER_ID"]))

# displaying the relevant answer
df_6 = combined_data.groupBy("PlayerID", "DefenderID") \
                    .agg(first("player_name").alias("Player Name"), first("CLOSEST_DEFENDER").alias("Useless Player/Kuppa Paiyan")) \
                    .orderBy("PlayerID") \
                    .limit(10)

df_6.show()
