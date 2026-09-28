"""
generate_catalog.py

Generates a synthetic product catalog with varied description phrasing,
rather than one fill-in-the-blank template repeated for every product.
Near-identical descriptions cause dense embeddings to cluster too tightly,
losing discriminative power between attributes like "navy" vs "olive" --
using multiple sentence templates and varied vocabulary gives embeddings
more signal to distinguish products on.

Synthetic data, not sourced from a real catalog -- keeps the eval focused on
retrieval methodology rather than real-data cleaning concerns.
"""

import json
import random

random.seed(42)

CATEGORIES = ["tops", "bottoms", "dresses", "outerwear", "footwear", "accessories"]
STYLES = ["casual", "formal", "athleisure", "streetwear", "minimalist"]
COLORS = ["black", "white", "navy", "beige", "olive", "grey", "burgundy", "rust", "sage", "cream"]

MATERIALS = ["cotton", "linen", "wool blend", "recycled polyester", "denim", "silk blend", "jersey knit"]
OCCASIONS = {
    "casual": ["a weekend brunch", "running errands", "a relaxed day out", "coffee with friends"],
    "formal": ["an evening event", "a business dinner", "a black-tie occasion", "a formal ceremony"],
    "athleisure": ["a gym session", "a morning run", "yoga class", "an active weekend"],
    "streetwear": ["a night out", "a concert", "hanging out downtown", "a casual meetup"],
    "minimalist": ["everyday wear", "the office", "a clean, understated look", "versatile styling"],
}
FIT_NOTES = ["a relaxed fit", "a tailored silhouette", "true to size", "a slightly oversized cut", "a fitted shape"]

DESCRIPTION_TEMPLATES = [
    "Crafted from {material}, this {style} {category_singular} is designed for {occasion}. Features {fit}.",
    "A {style} take on the classic {category_singular}, made with {material} and {fit}. Perfect for {occasion}.",
    "This {color} {category_singular} brings {style} styling to {occasion}. Made with {material}, offering {fit}.",
    "Effortless {style} style meets comfort in this {material} {category_singular}. Ideal for {occasion} with {fit}.",
    "Designed for {occasion}, this {category_singular} combines {material} construction with {fit} and a {style} edge.",
]

NAME_ADJECTIVES = {
    "casual": ["Everyday", "Relaxed", "Weekend", "Easy"],
    "formal": ["Refined", "Tailored", "Classic", "Elevated"],
    "athleisure": ["Active", "Dynamic", "Flex", "Motion"],
    "streetwear": ["Urban", "Bold", "Edge", "Rebel"],
    "minimalist": ["Essential", "Clean", "Pure", "Simple"],
}

CATEGORY_SINGULAR = {
    "tops": "top", "bottoms": "pair of bottoms", "dresses": "dress",
    "outerwear": "jacket", "footwear": "pair of shoes", "accessories": "accessory piece",
}


def build_description(category, style, color):
    material = random.choice(MATERIALS)
    occasion = random.choice(OCCASIONS[style])
    fit = random.choice(FIT_NOTES)
    category_singular = CATEGORY_SINGULAR[category]
    template = random.choice(DESCRIPTION_TEMPLATES)
    return template.format(
        material=material, style=style, occasion=occasion,
        fit=fit, category_singular=category_singular, color=color,
    )


def build_name(category, style, index):
    adjective = random.choice(NAME_ADJECTIVES[style])
    category_title = category[:-1].title() if category != "accessories" else "Accessory"
    return f"{adjective} {category_title} {index}"


def main(num_products=1000):
    catalog = []
    for i in range(num_products):
        category = random.choice(CATEGORIES)
        style = random.choice(STYLES)
        color = random.choice(COLORS)

        catalog.append({
            "product_id": f"P{i:05d}",
            "name": build_name(category, style, i),
            "category": category,
            "style_tag": style,
            "color": color,
            "price": round(random.uniform(15, 220), 2),
            "stock": random.randint(0, 200),
            "description": build_description(category, style, color),
        })

    with open("data/catalog_seed.json", "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Generated {len(catalog)} products with varied descriptions.")
    print(f"Sample: {catalog[0]['name']} - {catalog[0]['description']}")


if __name__ == "__main__":
    main(num_products=1000)
