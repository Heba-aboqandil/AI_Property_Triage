"""
pinecone_client.py
------------------
Persistent vector store for property listings.

Each time a listing is submitted in Tab 2 we:
  1. Build a short, searchable summary string from the listing + report.
  2. Convert that text to a 768-dim embedding using Gemini.
  3. Upsert the embedding + the full payload into a Pinecone index.

Tab 1 can then query the index by similarity to surface ANY past listing
the user asks about — not just the most recent one.

Free tier note:
  Pinecone Starter plan = 1 serverless index, 100k vectors. More than
  enough for a class project.

ENV VARS REQUIRED:
  PINECONE_API_KEY   – from https://app.pinecone.io
  GEMINI_API_KEY     – from https://aistudio.google.com/app/apikey
"""

import os
import json
import time
from typing import Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))  # Walks UP to find .env at project root

from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_NAME      = os.getenv("PINECONE_INDEX", "property-listings")
# gemini-embedding-001 is the current Gemini embedding model exposed by the
# google-genai SDK. The legacy "text-embedding-004" name now returns 404 on
# v1beta; use the new model name. We force output_dimensionality=768 so the
# vectors fit our Pinecone index dimension.
EMBED_MODEL     = "gemini-embedding-001"
EMBED_DIM       = 768
CLOUD           = "aws"
REGION          = "us-east-1"

# IMPORTANT: keep API keys in environment variables only — never commit them
# to the file. Set them once in PowerShell:
#   $env:PINECONE_API_KEY="pcsk_..."
#   $env:GEMINI_API_KEY="AIza..."
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Lazy initialisation
# ---------------------------------------------------------------------------
_pc     = None
_index  = None
_genai  = None


def _get_genai_client():
    """Return a configured Gemini client."""
    global _genai
    if _genai is None:
        _genai = genai.Client(api_key=GEMINI_API_KEY)
    return _genai


def _client():
    """Return a connected Pinecone index (creates the index if missing)."""
    global _pc, _index
    if _index is not None:
        return _index

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Set it in PowerShell with "
            "$env:PINECONE_API_KEY='pcsk_...'"
        )
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in PowerShell with "
            "$env:GEMINI_API_KEY='AIza...'"
        )

    # Initialise Gemini client
    _get_genai_client()

    _pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create the index on first run; idempotent on subsequent runs.
    existing = {idx["name"] for idx in _pc.list_indexes()}
    if INDEX_NAME not in existing:
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Wait until the index is ready before returning
        while not _pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    _index = _pc.Index(INDEX_NAME)
    return _index


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _embed(text: str) -> list[float]:
    """Convert a string to a 768-d embedding using Gemini (document mode)."""
    client = _get_genai_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="retrieval_document",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values


def _embed_query(text: str) -> list[float]:
    """Same as _embed but with task_type tuned for queries."""
    client = _get_genai_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="retrieval_query",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def upsert_listing(payload: dict, report: Optional[dict] = None) -> str:
    """
    Store a listing in Pinecone.

    Returns the vector ID used (also stored as submission_id in payload).
    """
    index = _client()

    # Build a searchable text representation
    desc       = (payload.get("description") or "").strip()
    agent      = payload.get("agent_name") or "Unknown"
    submission = payload.get("submission_id") or f"sub_{int(time.time())}"

    report_bits = []
    if report:
        for key in ("property_type", "location", "price_ils",
                    "num_rooms", "size_sqm", "key_features",
                    "rag_insight", "enrichment_notes"):
            v = report.get(key)
            if v not in (None, "", []):
                report_bits.append(f"{key}: {v}")

    searchable = "\n".join(filter(None, [
        f"Description: {desc}",
        f"Agent: {agent}",
        *report_bits,
    ]))

    vector = _embed(searchable)

    # Metadata: keep it small. Pinecone metadata has a size cap (~40KB).
    # We do NOT store base64 images here — too big. The raw images stay
    # in SQLite (chat_history.db) only.
    metadata = {
        "submission_id": submission,
        "agent_name":    agent,
        "description":   desc[:1500],
        "submitted_at":  payload.get("submitted_at", ""),
        "searchable":    searchable[:3000],
    }
    if report:
        # Only flat, JSON-safe primitives are allowed as metadata values
        for key in ("property_type", "routing_decision", "location"):
            v = report.get(key)
            if isinstance(v, (str, int, float, bool)):
                metadata[key] = v
        for key in ("price_ils", "num_rooms", "size_sqm", "confidence"):
            v = report.get(key)
            if isinstance(v, (int, float)):
                metadata[key] = v
        # Lists get stringified so Pinecone accepts them
        if isinstance(report.get("key_features"), list):
            metadata["key_features"] = ", ".join(str(x) for x in report["key_features"][:8])

    index.upsert(vectors=[{
        "id":       submission,
        "values":   vector,
        "metadata": metadata,
    }])

    return submission


def search_listings(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the top_k listings most similar to `query`.

    Returns a list of dicts with keys: id, score, metadata.
    """
    if not query or not query.strip():
        return []

    index = _client()
    vector = _embed_query(query)

    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    # Handle both dict and object response shapes (SDK version differences)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])

    results = []
    for match in matches:
        if isinstance(match, dict):
            results.append({
                "id":       match.get("id"),
                "score":    float(match.get("score", 0)),
                "metadata": match.get("metadata") or {},
            })
        else:
            results.append({
                "id":       getattr(match, "id", None),
                "score":    float(getattr(match, "score", 0)),
                "metadata": dict(getattr(match, "metadata", {}) or {}),
            })
    return results


def list_all_listings(limit: int = 100) -> list[dict]:
    """
    Return ALL listings in the index, newest first. Used by the chat when
    the user asks broad questions like "show me my listings" or "compare
    my apartments" — semantic search isn't the right tool there because
    the query has no specific content to match against.

    Trick: we query with an embedding of a very generic phrase ("a property
    listing for sale") which is close to ALL listings in vector space, then
    return everything up to `limit`. This works reliably on the free tier
    where `index.list()` isn't always available.
    """
    index = _client()
    results = []

    # Generic embedding — close to every property listing in vector space
    try:
        generic_vec = _embed_query("a property listing for sale")
    except Exception:
        # Fallback: zero vector (some SDK versions accept this)
        generic_vec = [0.0] * EMBED_DIM

    try:
        res = index.query(
            vector=generic_vec,
            top_k=min(limit, 100),
            include_metadata=True,
        )
        # Handle both dict and object response shapes
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        for match in matches:
            if isinstance(match, dict):
                results.append({
                    "id":       match.get("id"),
                    "score":    float(match.get("score", 0)),
                    "metadata": match.get("metadata") or {},
                })
            else:
                results.append({
                    "id":       getattr(match, "id", None),
                    "score":    float(getattr(match, "score", 0)),
                    "metadata": dict(getattr(match, "metadata", {}) or {}),
                })
    except Exception:
        return []

    # Sort newest first by submitted_at if present
    results.sort(
        key=lambda r: (r.get("metadata") or {}).get("submitted_at", ""),
        reverse=True,
    )
    return results


def count_listings() -> int:
    """Total number of listings stored. Useful for the sidebar badge."""
    try:
        stats = _client().describe_index_stats()
        return int(stats.get("total_vector_count", 0))
    except Exception:
        return 0


def health_check() -> bool:
    """True if Pinecone is reachable and the index is ready."""
    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        return False
    try:
        _client().describe_index_stats()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Quick CLI for debugging:
#     python pinecone_client.py count
#     python pinecone_client.py search "villa in Tel Aviv"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pinecone_client.py [count|search QUERY|health]")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "count":
        print(f"Listings in index: {count_listings()}")
    elif cmd == "search" and len(sys.argv) >= 3:
        q = " ".join(sys.argv[2:])
        for r in search_listings(q, top_k=5):
            print(json.dumps(r, ensure_ascii=False, indent=2))
    elif cmd == "health":
        print("OK" if health_check() else "NOT REACHABLE")
    else:
        print("Unknown command")