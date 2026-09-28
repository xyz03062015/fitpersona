"""
evaluate_retrieval.py

Measures retrieval quality (Recall@K, Precision@K) against a small
hand-labeled query set and logs the results to MLflow.

The labeled set is small (10-20 queries with known-correct product IDs) --
enough to catch regressions and compare fusion weights, not benchmark-scale.
"""

import json
import mlflow
from retrieval.hybrid_search import HybridSearcher

# Hand-labeled test set: query -> set of product_ids considered a correct
# match. Built by manually checking the catalog for products that genuinely
# satisfy each query's category/style/color criteria.
LABELED_QUERIES_PATH = "rag/labeled_queries.json"


def load_labeled_queries():
    with open(LABELED_QUERIES_PATH, "r") as f:
        return json.load(f)


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k_ids = set(retrieved_ids[:k])
    if not relevant_ids:
        return None
    hits = len(top_k_ids & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return None
    hits = sum(1 for pid in top_k_ids if pid in relevant_ids)
    return hits / len(top_k_ids)


def main():
    searcher = HybridSearcher()
    labeled_queries = load_labeled_queries()

    mlflow.set_experiment("fitpersona_retrieval_eval")

    with mlflow.start_run(run_name="hybrid_search_recall_eval"):
        mlflow.log_param("dense_weight", 0.6)
        mlflow.log_param("bm25_weight", 0.4)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        recalls_at_3 = []
        recalls_at_5 = []
        precisions_at_5 = []

        for item in labeled_queries:
            query = item["query"]
            relevant_ids = set(item["relevant_product_ids"])

            results = searcher.search(query, top_k=5)
            retrieved_ids = [r["product_id"] for r in results]

            r3 = recall_at_k(retrieved_ids, relevant_ids, 3)
            r5 = recall_at_k(retrieved_ids, relevant_ids, 5)
            p5 = precision_at_k(retrieved_ids, relevant_ids, 5)

            if r3 is not None:
                recalls_at_3.append(r3)
            if r5 is not None:
                recalls_at_5.append(r5)
            if p5 is not None:
                precisions_at_5.append(p5)

            print(f"Query: '{query}' ({len(relevant_ids)} relevant products)")
            print(f"  Retrieved: {retrieved_ids}")
            print(f"  Recall@3: {r3:.2f}, Recall@5: {r5:.2f}, Precision@5: {p5:.2f}\n")

        avg_recall_3 = sum(recalls_at_3) / len(recalls_at_3) if recalls_at_3 else 0
        avg_recall_5 = sum(recalls_at_5) / len(recalls_at_5) if recalls_at_5 else 0
        avg_precision_5 = sum(precisions_at_5) / len(precisions_at_5) if precisions_at_5 else 0

        mlflow.log_metric("avg_recall_at_3", avg_recall_3)
        mlflow.log_metric("avg_recall_at_5", avg_recall_5)
        mlflow.log_metric("avg_precision_at_5", avg_precision_5)
        mlflow.log_metric("num_queries", len(labeled_queries))

        print(f"\n=== Overall ===")
        print(f"Avg Recall@3: {avg_recall_3:.3f}")
        print(f"Avg Recall@5: {avg_recall_5:.3f}")
        print(f"Avg Precision@5: {avg_precision_5:.3f}")
        print(f"\nLogged to MLflow. Run 'mlflow ui' to view the dashboard.")


if __name__ == "__main__":
    main()
