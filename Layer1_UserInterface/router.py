"""
Three-route chat router for the AI Property Triage assistant.

Routes:
- pinecone   → user's own saved listings (memory recall)
- rag        → ChromaDB market comparison (similar listings from teammate's DB)
- perplexity → live web data (rates, prices, exchange)
- groq       → general real estate knowledge (default)
"""
import re

# ============================================================
# Pinecone — user's personal saved listings
# ============================================================
PINECONE_PATTERNS = [
    r"\bmy (saved |uploaded |submitted )?listings?\b",
    r"\bshow me my\b",
    r"\bsearch (my|all my) (listings?|properties|uploads?)\b",
    r"\bmy (listing|property) history\b",
    r"\ball my (listings?|properties|uploads?)\b",
    r"\bproperties? (i'?ve|i have) (uploaded|saved|submitted)\b",
    r"\bhow many (properties|listings) (have i|did i)\b",
    r"\blist (all )?my (listings?|properties)\b",
    r"\bcompare my (listings?|properties)\b",
    r"\bin my history\b",
]

# ============================================================
# RAG — ChromaDB market comparison
# ============================================================
RAG_PATTERNS = [
    r"\bsimilar listings?\b",
    r"\bsimilar properties\b",
    r"\bfrom (the|chroma) database\b",
    r"\bcompare with (the )?database\b",
    r"\bcompare with (the )?market\b",
    r"\bmarket comparison\b",
    r"\bfind similar (listings?|properties)\b",
    r"\bcompare this (property|listing)\b",
    r"\bshow similar (listings?|properties)\b",
    r"\bcomparable (properties|listings?)\b",
    r"\b\d+[- ]?bedroom\b.*\b(apartment|house|villa|property)\b",
    r"\b(apartment|house|villa|property)\b.*\b\d+[- ]?bedroom\b",
    r"\b(apartment|house|villa|property)\b.*\b(sea view|balcony|renovated|parking|kitchen)\b",
    r"\b(tel aviv|haifa|jerusalem|netanya|herzliya)\b.*\b(apartment|house|villa|property)\b",
]
# ============================================================
# Latest listing evaluation — user's most recent submitted listing
# ============================================================
LISTING_CONTEXT_PATTERNS = [
    r"\b(my|the)\s+(latest|last|recent|current)\s+(listing|property|apartment|house)\b",
    r"\b(latest|last|recent|current)\s+(listing|property|apartment|house)\s+(price|value|worth)\b",
    r"\bwhat do you think about my\s+(latest|last|recent|current)\s+(listing|property)",
    r"\b(is|was)\s+(my|the)\s+(latest|last|recent|current)\s+(listing|property).*\b(price|reasonable|fair|good|high|low)\b",
    r"\b(my|the)\s+(listing|property)\s+(price|value).*\b(reasonable|fair|good|high|low)\b",
]
# ============================================================
# Perplexity — live data
# ============================================================
PERPLEXITY_PATTERNS = [
    r"\bcurrent (price|rate|interest|mortgage|exchange)\b",
    r"\b(today'?s?|now|right now|currently)\b",
    r"\bexchange rate\b",
    r"\busd to ils\b",
    r"\bils to usd\b",
    r"\bmortgage rate\b",
    r"\binterest rate\b",
    r"\bbank of israel\b",
    r"\bproperty prices? in israel\b",
    r"\blatest (real estate|market|property)\b",
    r"\bthis (week|month|year) (in|prices)\b",
    r"\bhow much (does|is) .* cost\b",
]

# Compile regex once
_PINECONE_RE = re.compile("|".join(PINECONE_PATTERNS), re.IGNORECASE)
_RAG_RE = re.compile("|".join(RAG_PATTERNS), re.IGNORECASE)
_LISTING_CONTEXT_RE = re.compile("|".join(LISTING_CONTEXT_PATTERNS), re.IGNORECASE)
_PERPLEXITY_RE = re.compile("|".join(PERPLEXITY_PATTERNS), re.IGNORECASE)


def route(text: str, has_listing_context: bool = False) -> str:
    """
    Decide which backend to use for a user message.

    Order matters:
    1. Pinecone = user's own saved listings
    2. Latest listing price/value = Perplexity with latest listing context
    3. RAG = ChromaDB market comparison
    4. Perplexity = live data
    5. Groq = default fallback
    """
    if not text:
        return "groq"

    # 1. Personal memory → Pinecone
    if _PINECONE_RE.search(text):
        return "pinecone"

    # 2. Questions about the latest submitted listing price/value
    if has_listing_context and _LISTING_CONTEXT_RE.search(text):
        return "perplexity"

    # 3. Market comparison → RAG / ChromaDB
    if _RAG_RE.search(text):
        return "rag"

    # 4. Live data → Perplexity
    if _PERPLEXITY_RE.search(text):
        return "perplexity"

    # 5. Default → Groq
    return "groq"


def explain_route(text: str) -> str:
    """For debugging — shows why the router chose a route."""
    decision = route(text)
    reasons = {
        "pinecone":   "matched 'my listings' pattern → personal memory",
        "rag":        "matched 'similar listings' pattern → market comparison",
        "perplexity": "matched 'live data' pattern → web search",
        "groq":       "no specific pattern matched → general knowledge",
    }
    return f"[{decision}] {reasons[decision]}"