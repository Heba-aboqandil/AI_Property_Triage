"""
RAG client for ChromaDB queries via the EC2 RAG service.
Used by the chat assistant for market comparison queries.
"""
import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

RAG_URL = os.getenv("RAG_SERVICE_URL", "http://54.84.168.9:8001/query")
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "60"))


def query_rag(description: str) -> dict:
    """
    Query the EC2 RAG service for similar listings.

    Args:
        description: Property description (min 10 characters)

    Returns:
        dict with 'similar_listings' list and 'insight' string

    Raises:
        ValueError: if description is too short
        requests.exceptions.RequestException: on network/HTTP errors
    """
    if not description or len(description.strip()) < 10:
        raise ValueError("Description must be at least 10 characters long")

    response = requests.post(
        RAG_URL,
        json={"description": description.strip()},
        timeout=RAG_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def is_rag_available() -> bool:
    """Quick health check for the RAG service."""
    try:
        health_url = RAG_URL.replace("/query", "/health")
        r = requests.get(health_url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False