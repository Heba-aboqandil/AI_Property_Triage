import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException

from shared.schemas import AgentRequest, AgentResponse, HealthResponse
from app.graph import build_graph

app = FastAPI(
    title="LangGraph Agent Service",
    description="Stateful multi-step reasoning agent for complex property analysis questions.",
    version="1.0.0",
)

_graph = None


@app.on_event("startup")
async def startup():
    global _graph
    _graph = build_graph()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="langgraph_agent")


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(body: AgentRequest):
    """
    Runs the LangGraph agent on a complex listing question.
    The agent plans tool calls, executes them against the RAG and Image Analyser
    services, then synthesises a final answer.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised yet.")

    try:
        initial_state = {
            "query": body.query,
            "plan": None,
            "tools_to_call": None,
            "tool_outputs": None,
            "answer": None,
            "tools_used": None,
            "reasoning_steps": None,
        }
        final_state = await _graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return AgentResponse(
        answer=final_state.get("answer") or "No answer generated.",
        tools_used=final_state.get("tools_used") or [],
        reasoning_steps=final_state.get("reasoning_steps") or [],
    )
