"""
rag_chain.py

Retrieval-Augmented Generation chain: takes a natural-language style question,
retrieves matching products via hybrid search, and asks an LLM to generate a
grounded recommendation -- grounded meaning the LLM is instructed to only
reference the retrieved products, not invent items that don't exist in the
catalog.

Local development uses Ollama (runs on this machine, no API key needed). The
deployed demo can't run Ollama -- no persistent background process there --
so it switches to Gemini via env var instead. Same chain logic, different LLM
client underneath.

Set LLM_BACKEND=ollama (default) or LLM_BACKEND=gemini via environment
variable or a .env file to switch.
"""

import os
from dotenv import load_dotenv
from retrieval.hybrid_search import HybridSearcher

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

RECOMMENDATION_PROMPT = """You are a style assistant for FitPersona, a fashion recommendation service.

A user asked: "{query}"

Here are the products retrieved from the catalog that best match their request:
{products_text}

Using ONLY the products listed above (do not invent products that aren't listed), write a short,
friendly recommendation (2-4 sentences) explaining which product(s) fit their request and why.
Reference the actual product names and prices from the list above.
"""


def _format_products_for_prompt(products):
    lines = []
    for p in products:
        lines.append(
            f"- {p['name']} (ID: {p['product_id']}, ${p['price']}, "
            f"{p['color']} {p['category']}, style: {p['style_tag']})"
        )
    return "\n".join(lines)


def _get_llm():
    if LLM_BACKEND == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_BACKEND=gemini but GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        return ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
    else:
        from langchain_community.llms import Ollama
        return Ollama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))


class RecommendationChain:
    def __init__(self):
        print(f"Initializing RAG chain with LLM backend: {LLM_BACKEND}")
        self.searcher = HybridSearcher()
        self.llm = _get_llm()

    def recommend(self, query, top_k=5):
        products = self.searcher.search(query, top_k=top_k)

        if not products:
            return {
                "query": query,
                "products": [],
                "recommendation": "No matching products found for that request.",
            }

        products_text = _format_products_for_prompt(products)
        prompt = RECOMMENDATION_PROMPT.format(query=query, products_text=products_text)

        if LLM_BACKEND == "gemini":
            response = self.llm.invoke(prompt)
            recommendation_text = response.content
        else:
            recommendation_text = self.llm.invoke(prompt)

        return {
            "query": query,
            "products": products,
            "recommendation": recommendation_text.strip(),
        }


if __name__ == "__main__":
    chain = RecommendationChain()
    test_queries = [
        "I need a casual outfit for a rainy weekend, budget under $80",
        "formal navy dress for an event",
    ]
    for q in test_queries:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        result = chain.recommend(q, top_k=3)
        print(f"\nRetrieved {len(result['products'])} products:")
        for p in result["products"]:
            print(f"  - {p['name']} (${p['price']})")
        print(f"\nRecommendation:\n{result['recommendation']}")
