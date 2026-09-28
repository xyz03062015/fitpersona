"""
cdc_producer.py

Simulates a CDC (Change Data Capture) stream for FitPersona's product catalog.

Emits the same event contract a real CDC tool (e.g. Debezium) would produce --
insert/update/delete events with before/after state -- so the downstream
pipeline consumes it the same way it would consume real CDC. Swapping this for
a real Debezium connector later shouldn't require touching the consumer side.
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

TOPIC = "catalog.products.cdc"
BOOTSTRAP_SERVERS = "localhost:9092"


def load_seed_catalog(path="data/catalog_seed.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_change_event(product, op):
    before = None
    after = None

    if op == "c":
        after = product
    elif op == "u":
        before = dict(product)
        product["price"] = round(product["price"] * random.uniform(0.85, 1.15), 2)
        product["stock"] = max(0, product["stock"] + random.randint(-5, 5))
        after = product
    elif op == "d":
        before = product
        after = None

    return {
        "event_id": str(uuid.uuid4()),
        "source": {"table": "products", "db": "fitpersona_catalog"},
        "op": op,
        "ts_ms": int(time.time() * 1000),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "before": before,
        "after": after,
    }


def run(rate_per_sec=5, duration_sec=None):
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )

    catalog = load_seed_catalog()
    print(f"Loaded {len(catalog)} seed products. Producing to '{TOPIC}'...")

    start = time.time()
    sent = 0

    try:
        while True:
            product = random.choice(catalog)
            op = random.choices(["c", "u", "d"], weights=[0.1, 0.85, 0.05])[0]
            event = make_change_event(product, op)

            producer.send(TOPIC, key=product["product_id"], value=event)
            sent += 1

            if sent % 20 == 0:
                print(f"[{sent}] sent op={op} product_id={product['product_id']}")

            time.sleep(1.0 / rate_per_sec)

            if duration_sec and (time.time() - start) > duration_sec:
                break
    except KeyboardInterrupt:
        print("\nStopping producer.")
    finally:
        producer.flush()
        producer.close()
        print(f"Total events sent: {sent}")


if __name__ == "__main__":
    run()
