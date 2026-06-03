import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException

from shared.schemas import RAGRequest, RAGResponse, SimilarListing, HealthResponse
from app.retriever import retrieve_similar
from app.chain import generate_insight

app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation for property listing similarity and insight.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="rag_service")


@app.post("/query", response_model=RAGResponse)
async def query(body: RAGRequest):
    """
    Embeds the incoming property description, retrieves the top-3 most similar
    listings from ChromaDB, then generates a short insight using Llama.cpp.
    """
    try:
        listings = retrieve_similar(body.description)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not listings:
        raise HTTPException(status_code=404, detail="Vector store is empty. Run seed_listings.py first.")

    try:
        insight = await generate_insight(body.description, listings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {exc}")

    return RAGResponse(
        similar_listings=[SimilarListing(**l) for l in listings],
        insight=insight,
    )
