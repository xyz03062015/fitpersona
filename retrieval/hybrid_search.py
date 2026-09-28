"""
hybrid_search.py

Combines BM25 (lexical/keyword) retrieval with Qdrant dense-vector retrieval,
then merges and re-ranks results.

Dense search catches semantic matches ("something warm for winter" -> wool
coat) even with no shared words. BM25 catches exact/lexical matches -- a
specific product name, an exact color or size token -- that embeddings can
blur across near neighbors. Fusing both (weighted score sum) covers more
cases than either alone. Learned-sparse methods (SPLADE, Qdrant's native
sparse vectors) are a possible upgrade over BM25 here.

Neither BM25 nor dense embeddings understand numeric constraints, so a price
ceiling ("under $80") is pulled out of the query with a regex and applied as
a Qdrant payload filter before scoring, rather than left for similarity
search to approximate.
"""

import json
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

CATALOG_PATH = "data/catalog_seed.json"
COLLECTION_NAME = "fitpersona_products"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Weight given to the dense-vector score vs. the BM25 score when fusing.
# Product descriptions here are templated, so dense embeddings cluster
# tightly and lose discriminative power between attributes like "navy" vs
# "olive"; BM25's exact-token matching stays sharp regardless, hence the
# higher weight. A catalog with more varied copy would likely shift this
# toward dense.
DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6

# Matches "under $80", "under 80", "below $50", "less than $100", "<$80"
PRICE_CEILING_PATTERN = re.compile(
    r"(?:under|below|less than|<)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)


def extract_price_ceiling(query):
    """Pulls a numeric price ceiling out of a natural-language query, if present."""
    match = PRICE_CEILING_PATTERN.search(query)
    if match:
        return float(match.group(1))
    return None


class HybridSearcher:
    def __init__(self):
        print("Loading catalog and embedding model for hybrid search...")
        with open(CATALOG_PATH, "r") as f:
            self.catalog = json.load(f)

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Build BM25 index over the same document text used for embeddings.
        tokenized_corpus = [
            self._tokenize(self._doc_text(p)) for p in self.catalog
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _doc_text(product):
        return (
            f"{product['name']}. Category: {product['category']}. "
            f"Style: {product['style_tag']}. Color: {product['color']}. "
            f"{product['description']}"
        )

    @staticmethod
    def _tokenize(text):
        return text.lower().replace(".", "").replace(",", "").split()

    def search(self, query, top_k=5):
        price_ceiling = extract_price_ceiling(query)

        qdrant_filter = None
        if price_ceiling is not None:
            qdrant_filter = Filter(
                must=[FieldCondition(key="price", range=Range(lte=price_ceiling))]
            )

        # Dense side: embed the query, search Qdrant (with price filter if any).
        query_vector = self.model.encode(query).tolist()
        dense_hits = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k * 3,  # overfetch so fusion has more to work with
        )
        dense_scores = {hit.payload["product_id"]: hit.score for hit in dense_hits}
        dense_ids = set(dense_scores.keys())

        # Lexical side: BM25 over the same corpus, then apply the same price
        # ceiling manually (BM25 has no built-in filtering concept).
        tokenized_query = self._tokenize(query)
        bm25_scores_raw = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
        bm25_scores = {}
        for i in range(len(self.catalog)):
            product = self.catalog[i]
            if price_ceiling is not None and product["price"] > price_ceiling:
                continue
            bm25_scores[product["product_id"]] = bm25_scores_raw[i] / max_bm25

        # Fuse: every product that appears in either result set (post-filter)
        # gets a combined score. Products missing from one side get 0 there.
        all_ids = dense_ids | set(bm25_scores.keys())
        fused = []
        for pid in all_ids:
            d = dense_scores.get(pid, 0.0)
            b = bm25_scores.get(pid, 0.0)
            combined = DENSE_WEIGHT * d + BM25_WEIGHT * b
            fused.append((pid, combined, d, b))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:top_k]

        catalog_by_id = {p["product_id"]: p for p in self.catalog}
        results = []
        for pid, combined, d, b in top:
            product = catalog_by_id[pid]
            results.append({**product, "score": combined, "dense_score": d, "bm25_score": b})

        return results


if __name__ == "__main__":
    searcher = HybridSearcher()
    test_queries = [
        "casual weekend outfit under $80",
        "formal navy dress",
        "streetwear footwear under $50",
    ]
    for q in test_queries:
        ceiling = extract_price_ceiling(q)
        print(f"\n=== Query: '{q}' (price ceiling: {ceiling}) ===")
        for r in searcher.search(q, top_k=3):
            print(
                f"  {r['product_id']} | {r['name']} | ${r['price']} | "
                f"score={r['score']:.3f} (dense={r['dense_score']:.3f}, bm25={r['bm25_score']:.3f})"
            )
