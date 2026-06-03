"""
AgentState — the single typed dict that flows through every LangGraph node.

All fields are Optional so nodes can safely read state without knowing
which prior node has already populated it.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    query: str

    # Planner output
    plan: Optional[str]            # free-text plan produced by the planner node
    tools_to_call: Optional[list]  # list of {"tool": str, "args": dict}

    # Tool execution output
    tool_outputs: Optional[list]   # list of {"tool": str, "result": dict}

    # Final output
    answer: Optional[str]
    tools_used: Optional[list[str]]
    reasoning_steps: Optional[list[str]]
