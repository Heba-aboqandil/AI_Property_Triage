"""
sonar_client.py
---------------
Streaming Perplexity Sonar client for the Tab 1 chat.

Used by the router when the user's message needs live web data. Has its own
system prompt (separate from the Groq prompt) — a small, focused brief that
keeps Sonar's answer real-estate-scoped, concise, and grounded in sources.
"""

import os
import requests
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))  # Walks UP to find .env at project root

PPLX_MODEL    = os.getenv("PPLX_MODEL", "sonar")
PPLX_API_KEY  = os.getenv("PPLX_API_KEY", "").strip()
PPLX_ENDPOINT = "https://api.perplexity.ai/chat/completions"

# ---------------------------------------------------------------------------
# Sonar system prompt — Iteration 1
#
# Design notes:
# - Mirrors the Groq prompt's role/topic boundaries so behaviour stays
#   consistent across the two backends (the user shouldn't be able to tell
#   which one answered).
# - Tells Sonar to cite sources inline (Sonar already does this by default,
#   but the explicit instruction makes citations more consistent).
# - Caps length so the chat UI stays clean.
# - Allows specific numbers ONLY when Sonar found them on the web (Sonar's
#   whole purpose), unlike the Groq prompt which forbids inventing them.
# ---------------------------------------------------------------------------
SONAR_SYSTEM_PROMPT = """You are a real-estate market analyst answering a property agent's
question using LIVE web search results.

Role and scope:
- Help listing agents with real-estate questions that need current information:
  prices, mortgage rates, market trends, recent regulations, comparable sales.
- Currency exchange questions (e.g. USD to ILS) are in scope: many Israeli
  property listings are priced in USD or EUR but paid in ILS, so agents and
  buyers need the live FX rate to compare offers. Answer these directly with
  the current rate and, when natural, a one-line note tying it to property
  context (e.g. "useful when comparing a USD-priced Tel Aviv listing").
- For other off-topic questions (cooking, jokes, politics, etc.), reply:
  "I can only help with real estate topics. Could I help you with a property-related question?"

Output rules (STRICT):
- Maximum 5 sentences. ONE paragraph. NO bulleted or numbered lists.
- Cite sources inline using short names (e.g. "according to Yad2", "Madlan reports").
- Always include at least one concrete number when the search results contain one.
- If the search returned no relevant results, say so honestly — DO NOT invent numbers.
- Do NOT add a "Sources:" list at the end; weave citations into the prose.
- Do NOT start with "Based on the search results" or similar meta-commentary.

Professional limits:
- Do not provide legal advice; suggest consulting a licensed attorney for legal questions.
- Do not guarantee financial returns or price appreciation.
- For very specific personal situations, recommend the agent verify with a local professional.
"""


def stream_sonar(user_message: str, history: list[dict] | None = None,
                 listing_context: str | None = None):
    """
    Stream a Perplexity Sonar response chunk-by-chunk.

    Args:
        user_message: the user's latest message.
        history: prior conversation turns (list of {role, content}). Optional —
            Sonar usually answers stand-alone, but recent context helps with
            follow-ups like "what about Jerusalem?".
        listing_context: optional plain-text summary of the user's submitted
            listing so Sonar can evaluate it ("is this price reasonable?").

    Yields:
        Content chunks (strings) as Sonar streams them back.
    """
    if not PPLX_API_KEY:
        yield "[Sonar error: PPLX_API_KEY is not set. " \
              "Set it in PowerShell with $env:PPLX_API_KEY='pplx-...']"
        return

    # Build a single combined system message — Perplexity prefers ONE system
    # message at the start, not multiple stacked ones.
    full_system = SONAR_SYSTEM_PROMPT
    if listing_context:
        full_system += (
            "\n\nThe user has submitted this listing. When relevant, "
            "compare it with current market data from your web search.\n"
            f"--- USER'S LISTING ---\n{listing_context}\n--- END LISTING ---"
        )

    messages = [{"role": "system", "content": full_system}]

    # Sonar works best as a stateless web-search call: send the system prompt
    # + the current question only. Including chat history confuses it (the
    # earlier turns are answers from Groq about *other* topics) and sometimes
    # triggers 400 Bad Request when those turns contain error text or refusals.
    # Trade-off: follow-up questions like "what about Jerusalem?" lose context,
    # but accuracy on the main question is far more important.
    messages.append({"role": "user", "content": user_message})

    body = {
        "model": PPLX_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        with requests.post(PPLX_ENDPOINT, json=body, headers=headers,
                           stream=True, timeout=60) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                # Perplexity uses Server-Sent Events: "data: { ... }"
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Standard OpenAI-style streaming shape
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield content
    except requests.HTTPError as e:
        msg = ""
        try:
            msg = e.response.text[:200]
        except Exception:
            pass
        yield f"\n[Sonar HTTP error: {e}. {msg}]"
    except Exception as e:
        yield f"\n[Sonar error: {e}]"


def health_check() -> bool:
    """Return True if the Sonar endpoint is reachable and the key works."""
    if not PPLX_API_KEY:
        return False
    try:
        r = requests.post(
            PPLX_ENDPOINT,
            json={
                "model": PPLX_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {PPLX_API_KEY}"},
            timeout=8,
        )
        return r.status_code < 500
    except Exception:
        return False