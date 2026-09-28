"""
bronze_stream.py

Spark Structured Streaming job: consumes CDC events from Kafka and writes them
to a Delta Lake "bronze" table -- the raw, minimally-processed landing zone
(medallion architecture; silver/gold layers conflate and aggregate downstream).

Checkpointing tracks which Kafka offsets have been processed, so a crash/
restart resumes from where it left off instead of reprocessing or skipping.

Kafka's at-least-once delivery means a producer retry can send the same event
twice, so we dedupe on event_id within each micro-batch and MERGE (upsert) on
event_id when writing to bronze, so a replayed message doesn't create a
duplicate row.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, LongType
)
from delta import configure_spark_with_delta_pip

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "catalog.products.cdc"
BRONZE_PATH = "data/delta/bronze/catalog_events"
CHECKPOINT_PATH = "checkpoints/bronze_catalog_events"

product_schema = StructType([
    StructField("product_id", StringType()),
    StructField("name", StringType()),
    StructField("category", StringType()),
    StructField("style_tag", StringType()),
    StructField("color", StringType()),
    StructField("price", DoubleType()),
    StructField("stock", IntegerType()),
    StructField("description", StringType()),
])

event_schema = StructType([
    StructField("event_id", StringType()),
    StructField("source", StructType([
        StructField("table", StringType()),
        StructField("db", StringType()),
    ])),
    StructField("op", StringType()),
    StructField("ts_ms", LongType()),
    StructField("produced_at", StringType()),
    StructField("before", product_schema),
    StructField("after", product_schema),
])


def build_spark():
    builder = (
        SparkSession.builder.appName("FitPersonaBronzeStream")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
                .config("spark.sql.shuffle.partitions", "4")
    )
    return configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"],
    ).getOrCreate()


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    deduped = batch_df.dropDuplicates(["event_id"])

    from delta.tables import DeltaTable
    spark = deduped.sparkSession

    if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
        bronze = DeltaTable.forPath(spark, BRONZE_PATH)
        (
            bronze.alias("t")
            .merge(deduped.alias("s"), "t.event_id = s.event_id")
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        deduped.write.format("delta").mode("overwrite").save(BRONZE_PATH)

    print(f"[batch {batch_id}] wrote {deduped.count()} deduped events to bronze")


def run():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), event_schema).alias("event"))
        .select("event.*")
        .withColumn("ingested_at", current_timestamp())
    )

    query = (
        parsed.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print("Streaming to bronze Delta table. Press Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    run()
