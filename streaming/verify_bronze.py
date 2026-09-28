
"""
verify_bronze.py

Quick sanity-check: reads the bronze Delta table and prints row counts and a
sample. Run in a SEPARATE terminal while bronze_stream.py is running (or after
stopping it).
"""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

BRONZE_PATH = "data/delta/bronze/catalog_events"


def main():
    builder = (
        SparkSession.builder.appName("FitPersonaVerifyBronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.format("delta").load(BRONZE_PATH)

    print(f"\nTotal rows in bronze: {df.count()}")
    print("\nOp type breakdown:")
    df.groupBy("op").count().show()

    print("\nSample rows:")
    df.select("event_id", "op", "after.product_id", "after.price", "ingested_at").show(
        10, truncate=False
    )


if __name__ == "__main__":
    main()
