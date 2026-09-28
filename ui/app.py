"""
app.py

Streamlit demo UI for FitPersona. Calls the FastAPI backend's /recommend
endpoint and displays retrieved products plus the generated recommendation.

Run with: streamlit run ui/app.py
(Make sure the FastAPI backend is running first: uvicorn api.main:app)
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("FITPERSONA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FitPersona", page_icon="👗", layout="centered")

st.title("👗 FitPersona")
st.caption(
    "Streaming recommendation data platform — hybrid BM25 + dense retrieval, "
    "LangChain RAG, served live. [Architecture & roadmap in README]"
)

# Quick-prompt buttons for a guided feel without abandoning free-text RAG input.
quick_prompts = [
    "Casual weekend outfit under $80",
    "Formal outfit for an event",
    "Streetwear footwear",
]

st.write("**Try a quick prompt, or type your own:**")
cols = st.columns(len(quick_prompts))
selected_prompt = None
for col, prompt in zip(cols, quick_prompts):
    if col.button(prompt):
        selected_prompt = prompt

query = st.text_input(
    "What are you looking for?",
    value=selected_prompt or "",
    placeholder="e.g. a casual outfit for a rainy weekend, budget under $80",
)

if st.button("Get recommendation", type="primary") and query:
    with st.spinner("Searching catalog and generating recommendation..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query, "top_k": 5},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Recommendation")
            # Streamlit's markdown renderer treats text between two `$` signs
            # as LaTeX math notation -- since our recommendations naturally
            # contain multiple dollar amounts, that was swallowing whole
            # sentences into a math block (rendered in italics, no spaces).
            # Escaping the $ signs displays them as plain currency instead.
            safe_recommendation = data["recommendation"].replace("$", "\\$")
            st.write(safe_recommendation)

            st.subheader("Matching products")
            for p in data["products"]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"**{p['name']}**")
                        st.caption(
                            f"{p['category']} · {p['style_tag']} · {p['color']}"
                        )
                    with c2:
                        st.write(f"${p['price']:.2f}")
                        st.caption(f"score: {p['score']:.3f}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the FitPersona API. Make sure it's running: "
                "`uvicorn api.main:app`"
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

with st.expander("What's real vs. roadmap"):
    st.markdown("""
    **Built and verified:**
    - Kafka + simulated CDC events → Spark Structured Streaming → Delta Lake (bronze)
    - Verified checkpoint-based recovery: zero duplicate events after a forced crash/restart
    - Hybrid BM25 + dense retrieval (Qdrant), fusion weights tuned against a labeled query set
    - LangChain RAG chain, pluggable LLM backend (local Ollama / deployed Gemini)
    - Retrieval quality measured with Recall@K / Precision@K, tracked in MLflow

    **Roadmap (documented, not built):**
    - Real CDC automation (Debezium / Kafka Connect / Schema Registry)
    - Apache Iceberg lakehouse interoperability
    - Terraform / Kubernetes infrastructure provisioning
    """)
