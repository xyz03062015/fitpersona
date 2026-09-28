"""
build_labeled_queries.py

Generates rag/labeled_queries.json from the actual catalog data. Each labeled
query is built from category/style/color/price criteria, and "relevant"
product_ids are every product in the catalog that matches that criteria --
ground truth derived from the data itself rather than hand-typed IDs.

Queries combine multiple criteria (category + style + color + often a price
ceiling) rather than one loose attribute, since a single attribute like
style_tag alone can match 100+ products in this catalog, which caps Recall@K
near the relevant-set size regardless of retrieval quality. Narrowing the
relevant set to single digits keeps Recall@K meaningful.

Run this once (or whenever catalog_seed.json changes) before running
evaluate_retrieval.py.
"""

import json

CATALOG_PATH = "data/catalog_seed.json"
OUTPUT_PATH = "rag/labeled_queries.json"

# Each entry: a natural-language query, plus the filter criteria that define
# a "relevant" product for that query. Criteria are combined narrowly enough
# that each relevant set stays small (roughly 1-10 items).
QUERY_DEFINITIONS = [
    {
        "query": "casual navy top",
        "category": "tops", "style_tag": "casual", "color": "navy", "price_max": None,
    },
    {
        "query": "formal navy dress",
        "category": "dresses", "style_tag": "formal", "color": "navy", "price_max": None,
    },
    {
        "query": "streetwear black footwear",
        "category": "footwear", "style_tag": "streetwear", "color": "black", "price_max": None,
    },
    {
        "query": "athleisure tops under $50",
        "category": "tops", "style_tag": "athleisure", "color": None, "price_max": 50,
    },
    {
        "query": "minimalist black accessories",
        "category": "accessories", "style_tag": "minimalist", "color": "black", "price_max": None,
    },
    {
        "query": "casual bottoms in olive",
        "category": "bottoms", "style_tag": "casual", "color": "olive", "price_max": None,
    },
    {
        "query": "formal grey outerwear",
        "category": "outerwear", "style_tag": "formal", "color": "grey", "price_max": None,
    },
    {
        "query": "streetwear tops under $60",
        "category": "tops", "style_tag": "streetwear", "color": None, "price_max": 60,
    },
    {
        "query": "minimalist white dresses under $100",
        "category": "dresses", "style_tag": "minimalist", "color": "white", "price_max": 100,
    },
    {
        "query": "athleisure burgundy footwear",
        "category": "footwear", "style_tag": "athleisure", "color": "burgundy", "price_max": None,
    },
]


def matches(product, definition):
    if definition["category"] and product["category"] != definition["category"]:
        return False
    if definition["style_tag"] and product["style_tag"] != definition["style_tag"]:
        return False
    if definition["color"] and product["color"] != definition["color"]:
        return False
    if definition["price_max"] and product["price"] > definition["price_max"]:
        return False
    return True


def main():
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    labeled = []
    for definition in QUERY_DEFINITIONS:
        relevant_ids = [p["product_id"] for p in catalog if matches(p, definition)]
        if not relevant_ids:
            print(f"WARNING: '{definition['query']}' matched 0 products, skipping")
            continue
        labeled.append({
            "query": definition["query"],
            "relevant_product_ids": relevant_ids,
        })
        print(f"'{definition['query']}' -> {len(relevant_ids)} relevant products")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(labeled, f, indent=2)

    print(f"\nWrote {len(labeled)} labeled queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
