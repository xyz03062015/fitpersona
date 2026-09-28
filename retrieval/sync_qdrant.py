"""
sync_qdrant.py

Reads the silver (current-state) Delta table and syncs it into Qdrant --
only re-embeds and upserts products whose content actually changed since the
last sync, using a content hash. Deleted products get removed from Qdrant
entirely.

Each point's payload also carries embedding_model, embedding_model_version,
and catalog_version, so vectors produced by an older model version can be
identified later if the model changes.

Runs as a periodic loop reading a Delta table snapshot -- not a Kafka
consumer with a live subscription. A streaming sink is a possible next step.
"""

import hashlib
import json
import os
import time

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

SILVER_PATH = "data/delta/silver/catalog_current"
REGISTRY_PATH = "retrieval/embedding_registry.json"
COLLECTION_NAME = "fitpersona_products"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_VERSION = "v1"  # bump this manually if the model or its config changes
CATALOG_VERSION = "v2-rich-synthetic"  # bump when the catalog generation logic changes
SYNC_INTERVAL_SECONDS = 30


def content_hash(row):
    """Hash of the fields that actually affect the embedding + BM25 text."""
    raw = f"{row['name']}|{row['category']}|{row['style_tag']}|{row['color']}|{row['description']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def doc_text(row):
    return (
        f"{row['name']}. Category: {row['category']}. "
        f"Style: {row['style_tag']}. Color: {row['color']}. {row['description']}"
    )


def build_spark():
    builder = (
        SparkSession.builder.appName("FitPersonaQdrantSync")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def ensure_collection(client, vector_size):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'")


def sync_once(spark, model, qdrant, registry):
    df = spark.read.format("delta").load(SILVER_PATH)
    rows = [row.asDict() for row in df.collect()]

    to_upsert = []
    to_delete_ids = []
    unchanged_count = 0

    for row in rows:
        pid = row["product_id"]

        if row["is_deleted"]:
            if pid in registry:
                to_delete_ids.append(pid)
                del registry[pid]
            continue

        current_hash = content_hash(row)
        prior = registry.get(pid)

        if prior and prior["content_hash"] == current_hash:
            unchanged_count += 1
            continue  # skip re-embedding -- content hasn't changed

        to_upsert.append((row, current_hash))

    if to_upsert:
        texts = [doc_text(row) for row, _ in to_upsert]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

        points = []
        for (row, current_hash), vector in zip(to_upsert, embeddings):
            pid = row["product_id"]
            point_id = int(pid.replace("P", ""))  # Qdrant needs int or UUID ids
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "product_id": pid,
                        "name": row["name"],
                        "category": row["category"],
                        "style_tag": row["style_tag"],
                        "color": row["color"],
                        "price": row["price"],
                        "stock": row["stock"],
                        "description": row["description"],
                        "document_text": doc_text(row),
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_model_version": EMBEDDING_MODEL_VERSION,
                        "catalog_version": CATALOG_VERSION,
                    },
                )
            )
            registry[pid] = {
                "content_hash": current_hash,
                "embedding_model_version": EMBEDDING_MODEL_VERSION,
                "catalog_version": CATALOG_VERSION,
                "last_synced_at": time.time(),
            }

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    if to_delete_ids:
        point_ids = [int(pid.replace("P", "")) for pid in to_delete_ids]
        qdrant.delete(collection_name=COLLECTION_NAME, points_selector=point_ids)

    print(
        f"Sync: {len(to_upsert)} upserted, {len(to_delete_ids)} deleted, "
        f"{unchanged_count} unchanged (skipped re-embedding)"
    )
    return registry


def run(loop=True):
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qdrant, vector_size)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    registry = load_registry()

    while True:
        registry = sync_once(spark, model, qdrant, registry)
        save_registry(registry)

        if not loop:
            break

        print(f"Sleeping {SYNC_INTERVAL_SECONDS}s until next sync...")
        time.sleep(SYNC_INTERVAL_SECONDS)


if __name__ == "__main__":
    run(loop=True)
