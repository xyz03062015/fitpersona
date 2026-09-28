"""
main.py

FastAPI application wrapping the RAG recommendation chain as an HTTP API.
This is the serving layer -- what a real front end (Streamlit, a web app,
a mobile client) actually calls.

Run with: uvicorn api.main:app --reload
Docs available at http://localhost:8000/docs (FastAPI auto-generates this).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.rag_chain import RecommendationChain

app = FastAPI(
    title="FitPersona Recommendation API",
    description=(
        "Hybrid BM25 + dense retrieval over a fashion catalog, with LangChain "
        "RAG generating grounded style recommendations."
    ),
    version="0.1.0",
)

# Loaded once at startup, not per-request -- the embedding model and BM25
# index are expensive to initialize, so we keep one instance alive.
_chain = None


@app.on_event("startup")
def load_chain():
    global _chain
    _chain = RecommendationChain()


class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5


class ProductResult(BaseModel):
    product_id: str
    name: str
    category: str
    style_tag: str
    color: str
    price: float
    score: float


class RecommendResponse(BaseModel):
    query: str
    products: list[ProductResult]
    recommendation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if _chain is None:
        raise HTTPException(status_code=503, detail="Chain not initialized yet")

    result = _chain.recommend(request.query, top_k=request.top_k)

    products = [
        ProductResult(
            product_id=p["product_id"],
            name=p["name"],
            category=p["category"],
            style_tag=p["style_tag"],
            color=p["color"],
            price=p["price"],
            score=p["score"],
        )
        for p in result["products"]
    ]

    return RecommendResponse(
        query=result["query"],
        products=products,
        recommendation=result["recommendation"],
    )
