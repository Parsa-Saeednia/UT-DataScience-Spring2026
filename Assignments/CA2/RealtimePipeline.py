import shutil
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType, ArrayType
from pyspark.sql.functions import explode, col, sum, count, window, approx_count_distinct, struct, collect_set, from_json, avg, round, to_timestamp, to_json

shutil.rmtree('checkpoints', ignore_errors=True)

spark_version = pyspark.__version__
scala_ver = "2.13" if spark_version.startswith("4") else "2.12"
spark = SparkSession.builder \
    .appName("Ashpaz_RealTime") \
    .master("local[*]") \
    .config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_version}") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

item_schema = StructType([
    StructField("category", StringType(), True),
    StructField("food_item", StringType(), True),
    StructField("quantity", LongType(), True),
    StructField("unit_price", DoubleType(), True)
])

order_schema = StructType([
    StructField("cuisines", ArrayType(StringType()), True),
    StructField("items", ArrayType(item_schema), True),
    StructField("order_id", StringType(), True),
    StructField("order_price", DoubleType(), True),
    StructField("order_time", StringType(), True),
    StructField("phone_number", StringType(), True),
    StructField("request_online", BooleanType(), True),
    StructField("request_table", BooleanType(), True),
    StructField("restaurant_city", StringType(), True),
    StructField("restaurant_id", LongType(), True),
    StructField("restaurant_name", StringType(), True),
    StructField("user_id", StringType(), True)
])

raw_kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ashpaz.valid") \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream_df = raw_kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), order_schema).alias("data")) \
    .select("data.*")

stream_with_time = parsed_stream_df.withColumn("timestamp", to_timestamp(col("order_time")))

geo_fraud_df = stream_with_time.withWatermark("timestamp", "10 minutes") \
    .groupBy(col("user_id"), window(col("timestamp"), "30 minutes")) \
    .agg(approx_count_distinct("restaurant_city").alias("unique_cities"), collect_set("restaurant_city").alias("city_list")) \
    .filter(col("unique_cities") > 1) \
    .selectExpr("user_id", "window.start AS window_start", "window.end AS window_end", "'Geographical_Impossibility' AS fraud_type", "CAST(city_list AS STRING) AS details")

vel_fraud_df = stream_with_time.withWatermark("timestamp", "10 minutes") \
    .groupBy(col("user_id"), window(col("timestamp"), "60 minutes")) \
    .agg(count("order_id").alias("order_count")) \
    .filter(col("order_count") > 5) \
    .selectExpr("user_id", "window.start AS window_start", "window.end AS window_end", "'Velocity_Spam' AS fraud_type", "CAST(order_count AS STRING) AS details")

all_alerts_df = geo_fraud_df.union(vel_fraud_df)

all_alerts_df.selectExpr("user_id AS key", "to_json(struct(user_id, window_start, window_end, fraud_type, details)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "ashpaz.error_log") \
    .option("checkpointLocation", "checkpoints/fraud_alerts") \
    .outputMode("append") \
    .trigger(processingTime='1 seconds') \
    .start()

all_alerts_df.writeStream \
    .format("console") \
    .outputMode("update") \
    .option("truncate", "false") \
    .trigger(processingTime='1 seconds') \
    .start()

live_cuisine_df = parsed_stream_df.withColumn("cuisine", explode(col("cuisines"))) \
    .groupBy("cuisine") \
    .agg(round(sum("order_price"), 2).alias("total_revenue"), round(avg("order_price"), 2).alias("avg_order_value"))

live_cuisine_df.writeStream \
    .format("console") \
    .outputMode("complete") \
    .trigger(processingTime='1 seconds') \
    .start()

spark.streams.awaitAnyTermination()