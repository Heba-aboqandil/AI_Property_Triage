"""
HTTP tool wrappers that the LangGraph tool_executor_node calls.

These functions call the RAG Service and Image Analyser over HTTP.
URLs are read from environment variables so they can point to localhost
during local development and to EC2 IPs in production.

Tool descriptions are written for the planner LLM. Quality matters here —
vague descriptions cause the planner to skip tools or call the wrong one.
(Prompt engineering surface: LangGraph Tool Descriptions — Section 4.4 of the guideline)
"""

import os
import httpx

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8001")
IMAGE_ANALYSER_URL = os.getenv("IMAGE_ANALYSER_URL", "http://image_analyser:8002")
HTTP_TIMEOUT = float(os.getenv("TOOL_HTTP_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Tool metadata (name + description used by the planner prompt)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "rag_tool": {
        "name": "rag_tool",
        "description": (
            "Use this tool when you need to find similar past property listings from the internal "
            "knowledge base, or when you need context about how a described property compares to "
            "previously processed listings. Input: a property description string. "
            "Output: up to 3 similar listings with summaries and similarity scores, plus an AI-generated "
            "comparative insight. Use this for questions about market comparisons, renovation benchmarks "
            "from past listings, or pricing context from historical data."
        ),
        "input_schema": {"description": "string — the property listing text to compare against the knowledge base"},
    },
    "image_analyser_tool": {
        "name": "image_analyser_tool",
        "description": (
            "Use this tool when you need to classify a property image by room type or assess the "
            "physical condition of a room shown in a photograph. Input: a publicly accessible image URL. "
            "Output: room_type (one of: kitchen, bathroom, living_room, bedroom, exterior, other), "
            "a condition_score from 1 (poor) to 5 (excellent), and a confidence value. "
            "Use this for questions about which rooms need renovation, what the overall property condition "
            "looks like, or whether specific rooms are in acceptable condition."
        ),
        "input_schema": {"image_url": "string — a publicly accessible URL of the property image"},
    },
}


# ---------------------------------------------------------------------------
# Tool executor functions
# ---------------------------------------------------------------------------

async def call_rag_tool(description: str) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{RAG_SERVICE_URL}/query", json={"description": description})
        resp.raise_for_status()
        return resp.json()


async def call_image_analyser_tool(image_url: str) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{IMAGE_ANALYSER_URL}/analyse", json={"image_url": image_url})
        resp.raise_for_status()
        return resp.json()


TOOL_FUNCTIONS = {
    "rag_tool": call_rag_tool,
    "image_analyser_tool": call_image_analyser_tool,
}
