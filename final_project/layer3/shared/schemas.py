from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------

class RAGRequest(BaseModel):
    description: str = Field(..., min_length=10, description="Property listing text to query against the vector store")


class SimilarListing(BaseModel):
    id: str
    title: str
    summary: str
    score: float = Field(..., ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    similar_listings: List[SimilarListing]
    insight: str


# ---------------------------------------------------------------------------
# Image Analyser
# ---------------------------------------------------------------------------

class ImageRequest(BaseModel):
    image_url: str = Field(..., description="Publicly accessible URL of the property image")


class ImageResponse(BaseModel):
    room_type: str = Field(..., description="Predicted room type label, or 'uncertain' when confidence is below threshold")
    condition_score: Optional[int] = Field(None, ge=1, le=5, description="Condition score 1–5; null when uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Guardrails Service
# ---------------------------------------------------------------------------

class GuardrailsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to validate")


class GuardrailsResponse(BaseModel):
    passed: bool = Field(..., alias="pass", description="True if validation passed")
    reason: Optional[str] = Field(None, description="Reason for rejection when passed=False")
    safe_text: Optional[str] = Field(None, description="Sanitised text returned by output rail when passed=True")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# LangGraph Agent
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Complex multi-step question about the listing")


class AgentResponse(BaseModel):
    answer: str
    tools_used: List[str]
    reasoning_steps: List[str]


# ---------------------------------------------------------------------------
# Health check (shared across all services)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
