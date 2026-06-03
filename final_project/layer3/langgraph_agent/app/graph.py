"""
LangGraph StateGraph definition.

Flow:
    planner_node
        ↓
    [conditional] — if tools_to_call is non-empty → tool_executor_node → synthesiser_node
                  — if tools_to_call is empty     → synthesiser_node directly
        ↓
    END
"""

from functools import lru_cache

from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes import planner_node, tool_executor_node, synthesiser_node


def _should_execute_tools(state: AgentState) -> str:
    tools = state.get("tools_to_call") or []
    return "execute_tools" if tools else "synthesise"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesiser", synthesiser_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        _should_execute_tools,
        {
            "execute_tools": "tool_executor",
            "synthesise": "synthesiser",
        },
    )

    graph.add_edge("tool_executor", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()
