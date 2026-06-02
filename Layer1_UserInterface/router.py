"""
router.py
---------
Decides which backend handles a user message:

  - "groq"      → the local Groq/Llama assistant answers from its trained
                  knowledge (definitions, concepts, evergreen advice, greetings,
                  off-topic refusals).
  - "perplexity"→ the question needs live web data (current prices, today's
                  mortgage rates, recent news, market trends, "is my listing
                  fairly priced", etc.); answer via Perplexity Sonar.

The router is a fast, deterministic regex/keyword classifier — no LLM call.
This is the "router pattern" recommended by the instructor: the 8B Groq
model is unreliable at function calling, so we make the routing decision in
plain Python instead and forward the message to the right backend.

Behaviour is conservative: when in doubt, route to Groq (cheaper, faster).
Perplexity is only invoked when a clear "live data" signal is present.
"""

import re

# Words/phrases that strongly imply the question needs real-time web data
LIVE_DATA_PATTERNS = [
    # Time signals
    r"\b(current(ly)?|today|tonight|this (week|month|year)|right now|at the moment)\b",
    r"\b(latest|recent(ly)?|new(est)?|up[- ]?to[- ]?date)\b",
    r"\b(202[5-9]|203\d)\b",                 # explicit recent year
    r"\b(now|nowadays|these days)\b",
    # Market-data signals (numbers the model can't know offline)
    r"\b(price|prices|pricing|cost|costs|rate|rates)\b.*\b(today|now|current|202[5-9])\b",
    r"\b(mortgage|interest)\s+rate",
    r"\b(price\s*per\s*(sqm|m²|square\s*meter))\b",
    r"\b(average|median)\s+(price|rent|cost)\b",
    r"\b(market\s+(value|trend|forecast|outlook|update|report))\b",
    # Live-listings / comparable-sales signals
    r"\b(comparable|comp)\s+(sale|listing|propert)",
    r"\b(what.?s\s+selling|currently\s+listed|on\s+the\s+market)\b",
    # News signals
    r"\b(news|update|headline|announcement)\b",
    # Direct "search the web" intent
    r"\b(search|look\s*up|find\s+(out\s+)?online|web\s+search)\b",
    # Currency / FX — relevant for property transactions priced in different
    # currencies (e.g. USD-priced Tel Aviv listings paid in ILS)
    r"\b(usd|dollar|dollars)\s*(to|in|vs|\-|/)\s*(ils|shekel|shekels|nis)\b",
    r"\b(shekel|shekels|ils|nis)\s*(to|in|vs|\-|/)\s*(usd|dollar|dollars|euro|eur)\b",
    r"\b(exchange\s+rate|fx\s+rate|forex|currency\s+rate)\b",
    r"\bhow\s+much\s+is\s+(the\s+)?(dollar|euro|usd|eur|shekel)\b",
]

# Words that signal evaluating the user's submitted listing — also needs live
# market data to do a real comparison
LISTING_EVALUATION_PATTERNS = [
    r"\b(my|the|this)\s+(listing|property|apartment|house|villa)\b.*\b(price|value|worth|reasonable|fair|compare|market)\b",
    r"\b(is\s+(my|the|this)\s+\w+\s+price\s+(reasonable|fair|good|right|too\s+(high|low)))\b",
    r"\b(how\s+does\s+(my|the|this)\s+.+\s+compare)\b",
    r"\b(market\s+(value|price)\s+for\s+(my|the|this))\b",
]

# Words that signal the user wants to RETRIEVE a previously-submitted listing
# from the vector store (Pinecone), not the most recent one. These are
# "memory" questions: "find the studio I uploaded", "show me my villa",
# "what was that office listing again", "compare my two apartments", etc.
PINECONE_RETRIEVAL_PATTERNS = [
    # Plural / collection: "my listings", "all my properties"
    r"\b(all\s+)?my\s+(listings|properties|reports|submissions)\b",
    # Find / search / show + my/the + property noun
    r"\b(find|show|list|search|retrieve|recall|look\s*up|pull\s*up|bring\s*up)\b.*\b(my|the|a|an|that|those|previous|earlier|past|prior|old)\s+\w*\s*(listing|listings|property|properties|apartment|villa|house|studio|office|report)\b",
    # "That [thing] I uploaded/submitted/added"
    r"\b(that|those|the|a|an)\s+\w*\s*(listing|property|apartment|villa|house|studio|office|report)\b.*\b(i\s+(uploaded|submitted|added|posted|sent|put))\b",
    r"\b(i\s+(uploaded|submitted|added|posted))\b.*\b(listing|property|apartment|villa|house|studio|office)\b",
    # Time references with property: "yesterday's listing", "the one from last week"
    r"\b(yesterday|last\s+(week|hour|night)|earlier|before|previously|a\s+while\s+ago|hours?\s+ago)\b.*\b(listing|property|apartment|villa|house|studio|office|report|upload)\b",
    r"\b(listing|property|apartment|villa|house|studio|office|report)\b.*\b(yesterday|last\s+(week|hour|night)|earlier|before|previously|a\s+while\s+ago|hours?\s+ago)\b",
    # Comparing across multiple listings
    r"\bcompare\s+(my|the|all)\s+(listings|properties|reports|two|three)\b",
    r"\b(which|what)\s+(of\s+)?(my|the)\s+(listings|properties|reports)\b",
    # Direct recall: "what did I submit last", "remember the villa I posted"
    r"\b(what\s+(did|have)\s+i\s+(submit|upload|add|post))\b",
    r"\b(remember|recall)\b.*\b(listing|property|apartment|villa|house|studio|office|i\s+(uploaded|submitted|posted))\b",
    # Listings about a specific area/feature the user described before
    r"\b(the|that)\s+(villa|apartment|house|studio|office|property)\s+(in|with|near|on|at)\s+\w+",
]

# Pre-compile for speed
_LIVE_DATA_RE = re.compile("|".join(LIVE_DATA_PATTERNS), re.IGNORECASE)
_LISTING_EVAL_RE = re.compile("|".join(LISTING_EVALUATION_PATTERNS), re.IGNORECASE)
_PINECONE_RE = re.compile("|".join(PINECONE_RETRIEVAL_PATTERNS), re.IGNORECASE)

# A question must contain at least one of these to qualify for Perplexity.
# This guards against "current dollar rate?" type questions that have a
# time signal but aren't actually about real estate.
REAL_ESTATE_TERMS = [
    r"\b(property|properties|real[- ]?estate|housing|home|homes)\b",
    r"\b(apartment|apartments|flat|flats|condo|villa|house|houses)\b",
    r"\b(office|retail|industrial|commercial)\s+(space|building|property|listing)?",
    r"\b(rent|rental|rentals|lease|leasing|tenant|tenancy|landlord)\b",
    r"\b(mortgage|loan|down\s*payment|escrow|appraisal|appraiser)\b",
    r"\b(listing|listings|seller|buyer|broker|agent|realtor)\b",
    r"\b(neighbourhood|neighborhood|district|zone|zoning)\b",
    r"\b(sqm|m²|square\s*meter|square\s*foot|sq\.?\s*ft)\b",
    r"\b(price[- ]per[- ]sqm|price[- ]per[- ]square)\b",
    r"\b(market\s+(value|trend|forecast))\b",
    # Generic terms that, combined with a live-data signal, indicate a
    # real-estate market question even when no explicit property noun is
    # present (e.g. "current Tel Aviv prices", "rent in Jerusalem today").
    r"\b(price|prices|pricing|cost|costs)\b",
    # Currency questions are allowed in this real-estate context because
    # property prices in Israel are often quoted in USD/EUR but paid in ILS.
    # Agents and buyers routinely need the live FX rate to compare offers.
    r"\b(usd|dollar|dollars|euro|eur|shekel|shekels|ils|nis|exchange\s+rate|forex|fx)\b",
]
_REAL_ESTATE_TERMS_RE = re.compile("|".join(REAL_ESTATE_TERMS), re.IGNORECASE)


def route(user_message: str, has_listing_context: bool = False) -> str:
    """
    Decide which backend should handle `user_message`.

    Args:
        user_message: the latest user input text.
        has_listing_context: whether the user has a recent listing loaded
            in the session.

    Returns:
        "perplexity", "pinecone", or "groq".
    """
    if not user_message or not user_message.strip():
        return "groq"

    text = user_message.strip()

    # CHECK FIRST: retrieving a previously-submitted listing from vector store.
    # We check this before live-data because a question like "find that villa
    # I uploaded last week" matches both "last week" (live-data signal) and
    # "I uploaded" (retrieval signal). Retrieval wins.
    if _PINECONE_RE.search(text):
        return "pinecone"

    # GUARD: even if "current" / "today" appears, the question must also
    # contain a real-estate term — otherwise it might be off-topic (e.g.
    # "what's the dollar rate today?" mentions "today" but is a currency
    # question). Off-topic questions go to Groq, which refuses politely.
    has_re_term = _REAL_ESTATE_TERMS_RE.search(text) is not None

    # Strong signal: explicit live-data keywords + a real-estate term present
    if _LIVE_DATA_RE.search(text) and has_re_term:
        return "perplexity"

    # Listing-evaluation language, only when a listing is actually loaded
    if has_listing_context and _LISTING_EVAL_RE.search(text):
        return "perplexity"

    # Default: Groq handles it (definitions, greetings, generic advice, refusals)
    return "groq"


def explain_route(user_message: str, has_listing_context: bool = False) -> dict:
    """
    Debug helper. Returns the decision PLUS which pattern matched (if any).
    Useful for the demo video to show the router's reasoning live.
    """
    decision = route(user_message, has_listing_context)
    matched = None
    if decision == "perplexity":
        m = _LIVE_DATA_RE.search(user_message)
        if m:
            matched = ("live_data", m.group(0))
        else:
            m = _LISTING_EVAL_RE.search(user_message)
            if m:
                matched = ("listing_evaluation", m.group(0))
    return {"decision": decision, "matched": matched}