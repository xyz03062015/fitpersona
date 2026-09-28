# FitPersona

FitPersona is a portfolio project that implements an end-to-end streaming data
platform for a fashion product catalog: simulated CDC events flow through
Kafka and Spark Structured Streaming into a Delta Lake bronze/silver setup,
get synced into Qdrant with a versioned embedding pipeline, and are served
through a hybrid (BM25 + dense) retrieval layer and a RAG recommendation
chain behind a FastAPI + Streamlit front end. It exists to demonstrate the
full pipeline — CDC ingestion, stream processing, a lakehouse-style storage
layer, hybrid search, and a RAG serving layer — working together on one
system, not as a production service.

## Architecture

```
CDC event simulator -> Kafka -> Spark Structured Streaming -> Delta Lake (bronze)
                                                                     |
                                                                     v
                                                    Delta Lake (silver, current state)
                                                                     |
                                                                     v
                                           Versioned sync (content-hash) -> Qdrant
                                                                     |
                                                                     v
                                        Hybrid search (BM25 + dense) -> RAG chain
                                                                     |
                                                                     v
                                                    FastAPI -> Streamlit UI
```

- **producer/** simulates CDC events (insert/update/delete with before/after
  state) onto a Kafka topic.
- **streaming/** consumes from Kafka into a bronze Delta table (raw event
  log), then conflates bronze into a silver table (one row per product,
  current state).
- **retrieval/** syncs silver into Qdrant and runs hybrid BM25 + dense search
  over it.
- **rag/** builds a labeled eval set, runs a RAG chain over hybrid search
  results, and evaluates retrieval quality.
- **api/** and **ui/** are the FastAPI serving layer and Streamlit front end.

## What's built vs. what's roadmap

**Built:**
- Python CDC event simulator producing the same event contract Debezium
  would produce
- Spark Structured Streaming with checkpointed, idempotent MERGE writes on
  `event_id`
- Silver layer with event-time conflation (merges on `ts_ms`, not arrival
  order) and soft-delete handling (`is_deleted`, not physical removal)
- Content-hash-based versioned embedding sync to Qdrant — skips re-embedding
  products whose content hasn't changed
- Hybrid BM25 + dense retrieval with a regex-extracted price-ceiling filter
- RAG recommendation chain with a pluggable LLM backend (Ollama local or
  Gemini)
- FastAPI serving layer
- Streamlit UI
- MLflow-logged retrieval evaluation (Recall@K, Precision@K)

**Roadmap (explicitly not built):**
- Real CDC via Debezium / Kafka Connect / Schema Registry
- Apache Iceberg
- Terraform / Kubernetes provisioning
- A live Kafka-to-vector-store streaming sink — the current sync
  (`retrieval/sync_qdrant.py`) is a 30-second polling loop reading a Delta
  snapshot, not a streaming consumer
- Production-scale deployment

## Verified metrics

- Catalog: 1,000 products
- Bronze events: 7,299 total (687 creates, 6,227 updates, 385 deletes)
- Silver: 995 current-state rows, 45 soft-deleted (950 active)
- Qdrant sync: 950 points upserted on a full sync; versioned sync skips
  re-embedding unchanged products (by content hash) on subsequent runs
- Retrieval quality (10-query hand-labeled eval set, narrow multi-attribute
  queries): Recall@3 0.80, Recall@5 0.92, Precision@5 0.62 — logged to MLflow
- Crash recovery: verified by forcibly killing and restarting the bronze
  streaming job mid-run; zero duplicate `event_id`s after restart
  (checkpointed, idempotent MERGE on `event_id`)

## Scope

Single-machine, synthetic data, portfolio-scale. This is not a
production-scale deployment — see the roadmap above for what that would
require.

## Setup

Requires Docker and Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd docker
docker-compose up -d
docker ps   # confirm fitpersona-kafka, fitpersona-qdrant are running

docker exec -it fitpersona-kafka kafka-topics --create \
  --topic catalog.products.cdc \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 1

cd ..
cp .env.example .env   # fill in GOOGLE_API_KEY if using LLM_BACKEND=gemini
```

## Running

Run in order, each in its own terminal:

```bash
python producer/cdc_producer.py          # simulates CDC events onto Kafka
python streaming/bronze_stream.py        # Kafka -> Delta bronze
python streaming/silver_stream.py        # bronze -> Delta silver (current state)
python retrieval/sync_qdrant.py          # silver -> Qdrant (versioned sync)
uvicorn api.main:app --reload            # FastAPI serving layer
streamlit run ui/app.py                  # UI, calls the FastAPI backend
```

`retrieval/hybrid_search.py`, `rag/build_labeled_queries.py`, and
`rag/evaluate_retrieval.py` can be run standalone once Qdrant is populated.

## Tech stack

Kafka, Spark Structured Streaming, Delta Lake, Qdrant, sentence-transformers,
rank-bm25, LangChain, FastAPI, Streamlit, MLflow, Ollama / Gemini.
