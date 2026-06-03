"""
LangGraph node functions.

Three nodes:
  1. planner_node    — LLM reads the query and tool registry, decides which tools to call
  2. tool_executor_node — executes the planned tool calls via HTTP
  3. synthesiser_node   — LLM combines tool outputs into a final answer

The planner and synthesiser use an external LLM (Gemini or GPT-4o) because
Llama.cpp is already loaded in the RAG service on the same host and running two
large models simultaneously would exceed t3.large memory.
"""

import json
import os

from langchain_core.messages import SystemMessage, HumanMessage

from app.state import AgentState
from app.tools import TOOL_REGISTRY, TOOL_FUNCTIONS

# ---------------------------------------------------------------------------
# LLM setup — defaults to OpenAI GPT-4o; set AGENT_LLM_PROVIDER=gemini to switch
# ---------------------------------------------------------------------------

_LLM_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "openai").lower()
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
_GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "")


def _get_llm():
    if _LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=_GEMINI_KEY, temperature=0.1)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", openai_api_key=_OPENAI_KEY, temperature=0.1)


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = _get_llm()
    return _llm


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a senior property analyst assistant. You receive a complex question about \
a real estate property and decide which tools to call to answer it.

Available tools:
{tool_descriptions}

Instructions:
- Read the user question carefully.
- Decide which tools are needed. You may call zero, one, or both tools.
- Return a JSON object with two keys:
  "plan": a one-sentence description of your reasoning,
  "tools_to_call": a list of tool call objects, each with "tool" and "args" keys.
  - For rag_tool, the args object must use exactly this key: "description".
- For image_analyser_tool, the args object must use exactly this key: "image_url".

Example output:
{{
  "plan": "The question asks about room condition so I will call image_analyser_tool.",
  "tools_to_call": [
    {{"tool": "image_analyser_tool", "args": {{"image_url": "https://example.com/img.jpg"}}}}
  ]
}}

If no tools are needed, return an empty list for tools_to_call.
Output ONLY valid JSON. No markdown, no explanation outside the JSON object.
"""


def _build_tool_descriptions() -> str:
    parts = []
    for tool in TOOL_REGISTRY.values():
        parts.append(f"Tool name: {tool['name']}\nDescription: {tool['description']}\n")
    return "\n".join(parts)


async def planner_node(state: AgentState) -> AgentState:
    llm = get_llm()
    system_prompt = _PLANNER_SYSTEM.format(tool_descriptions=_build_tool_descriptions())
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]
    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        plan = parsed.get("plan", "")
        tools_to_call = parsed.get("tools_to_call", [])
    except json.JSONDecodeError:
        plan = "Failed to parse planner output."
        tools_to_call = []

    return {
        **state,
        "plan": plan,
        "tools_to_call": tools_to_call,
        "reasoning_steps": [f"Planner: {plan}"],
        "tools_used": [],
        "tool_outputs": [],
    }


# ---------------------------------------------------------------------------
# Tool executor node
# ---------------------------------------------------------------------------

async def tool_executor_node(state: AgentState) -> AgentState:
    tools_to_call = state.get("tools_to_call") or []
    tool_outputs = []
    tools_used = []
    steps = list(state.get("reasoning_steps") or [])

    for call in tools_to_call:
        tool_name = call.get("tool")
        args = call.get("args", {})
        if tool_name == "rag_tool":
            if "description" not in args:
                if "property_description" in args:
                    args["description"] = args.pop("property_description")
                elif "query" in args:
                    args["description"] = args.pop("query")
                elif "text" in args:
                    args["description"] = args.pop("text")

        # LLMs may produce "url" instead of "image_url".
        if tool_name == "image_analyser_tool":
            if "image_url" not in args:
                if "url" in args:
                    args["image_url"] = args.pop("url")
                elif "image" in args:
                    args["image_url"] = args.pop("image")

        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            steps.append(f"Tool executor: Unknown tool '{tool_name}' — skipped.")
            continue

        try:
            result = await fn(**args)
            tool_outputs.append({"tool": tool_name, "result": result})
            tools_used.append(tool_name)
            steps.append(f"Tool executor: Called '{tool_name}' successfully.")
        except Exception as exc:
            tool_outputs.append({"tool": tool_name, "result": {"error": str(exc)}})
            steps.append(f"Tool executor: '{tool_name}' raised an error — {exc}")

    return {
        **state,
        "tool_outputs": tool_outputs,
        "tools_used": tools_used,
        "reasoning_steps": steps,
    }


# ---------------------------------------------------------------------------
# Synthesiser node
# ---------------------------------------------------------------------------

_SYNTHESISER_SYSTEM = """\
You are a senior property analyst. You have been given a user question and the \
outputs of one or more tools that retrieved relevant information.

Your task:
- Write a clear, concise answer (3–8 sentences) that directly addresses the question.
- Use only the information present in the tool outputs. Do not invent facts.
- If a tool returned an error or no data, acknowledge the gap honestly.
- Do not recommend prices or provide legal advice.
"""

_SYNTHESISER_HUMAN = """\
User question: {query}

Tool outputs:
{tool_outputs}

Write the answer now.
"""


async def synthesiser_node(state: AgentState) -> AgentState:
    llm = get_llm()
    tool_outputs_text = json.dumps(state.get("tool_outputs") or [], indent=2)
    steps = list(state.get("reasoning_steps") or [])

    messages = [
        SystemMessage(content=_SYNTHESISER_SYSTEM),
        HumanMessage(content=_SYNTHESISER_HUMAN.format(
            query=state["query"],
            tool_outputs=tool_outputs_text,
        )),
    ]
    response = await llm.ainvoke(messages)
    answer = response.content.strip()
    steps.append("Synthesiser: Final answer generated.")

    return {
        **state,
        "answer": answer,
        "reasoning_steps": steps,
    }
