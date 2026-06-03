"""
ChromaDB vector store + HuggingFace sentence-transformer embeddings.

The persisted Chroma collection is loaded from CHROMA_PATH (default:
/service/chroma_db). The seed_listings.py script must be run before the
service starts to populate the collection.
"""

import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = os.getenv("CHROMA_PATH", "/service/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "property_listings"
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )


def retrieve_similar(description: str) -> list[dict]:
    """Return top-K similar listings as a list of dicts with score."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(description, k=TOP_K)
    listings = []
    for doc, score in results:
        listings.append({
            "id": doc.metadata.get("id", "unknown"),
            "title": doc.metadata.get("title", ""),
            "summary": doc.page_content,
            "score": round(float(score), 4),
        })
    return listings
