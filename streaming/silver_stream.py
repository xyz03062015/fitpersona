"""
silver_stream.py

Reads the bronze Delta table (an append-only CDC event log) and maintains a
SILVER table: one row per product_id holding its latest known state.

Kept as a separate streaming job from bronze_stream.py -- bronze just lands
raw events, silver conflates them into current state, so each job has one
responsibility.

Merges keeping whichever row has the higher ts_ms (event time), not whichever
arrived last in the micro-batch, since Kafka doesn't guarantee strict
ordering across partitions and a late-arriving but chronologically older
event shouldn't overwrite newer state.

Deleted products are kept with is_deleted=true rather than physically
removed, so the downstream Qdrant sync job has an explicit signal to remove
them from the search index without having to diff against bronze.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, row_number, current_timestamp
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip

BRONZE_PATH = "data/delta/bronze/catalog_events"
SILVER_PATH = "data/delta/silver/catalog_current"
CHECKPOINT_PATH = "checkpoints/silver_catalog_current"


def build_spark():
    builder = (
        SparkSession.builder.appName("FitPersonaSilverStream")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "4")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Derive a single product_id column (present on 'after' for
    # create/update, only on 'before' for delete).
    with_pid = batch_df.withColumn(
        "product_id", coalesce(col("after.product_id"), col("before.product_id"))
    )

    # Keep only the highest-ts_ms row per product_id within this micro-batch,
    # in case of out-of-order delivery within the same batch.
    window = Window.partitionBy("product_id").orderBy(col("ts_ms").desc())
    latest_per_product = (
        with_pid.withColumn("rn", row_number().over(window))
        .filter(col("rn") == 1)
        .drop("rn")
    )

    silver_rows = latest_per_product.select(
        col("product_id"),
        col("after.name").alias("name"),
        col("after.category").alias("category"),
        col("after.style_tag").alias("style_tag"),
        col("after.color").alias("color"),
        col("after.price").alias("price"),
        col("after.stock").alias("stock"),
        col("after.description").alias("description"),
        col("op"),
        col("ts_ms"),
        (col("op") == "d").alias("is_deleted"),
    ).withColumn("silver_updated_at", current_timestamp())

    from delta.tables import DeltaTable
    spark = silver_rows.sparkSession

    if DeltaTable.isDeltaTable(spark, SILVER_PATH):
        silver = DeltaTable.forPath(spark, SILVER_PATH)
        (
            silver.alias("t")
            .merge(silver_rows.alias("s"), "t.product_id = s.product_id")
            .whenMatchedUpdateAll("s.ts_ms > t.ts_ms")  # only overwrite with newer events
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        silver_rows.write.format("delta").mode("overwrite").save(SILVER_PATH)

    print(f"[silver batch {batch_id}] merged {silver_rows.count()} product states")


def run():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    bronze_stream = spark.readStream.format("delta").load(BRONZE_PATH)

    query = (
        bronze_stream.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="15 seconds")
        .start()
    )

    print("Streaming bronze -> silver (current-state table). Press Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    run()
