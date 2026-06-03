"""
LCEL RAG chain:  retrieved_context + query  →  Llama.cpp  →  insight string.

Prompt engineering surface #3 (Prompt Engineering Log).
The prompt instructs the model to:
  - cite which retrieved listing it draws from (by ID)
  - stay strictly within the retrieved context
  - not fabricate details absent from the documents
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.llm import get_llm

_SYSTEM = """\
You are a senior real estate analyst assistant. You have been given a set of \
similar past property listings retrieved from an internal knowledge base. \
Your task is to write a short insight (3–5 sentences) that helps the listing \
agent understand how the new property compares to past listings.

Rules you must follow:
1. Use only facts from:
   - the new property description, and
   - the retrieved listings.
2. Do not invent prices, features, locations, condition, size, or legal claims.
3. Never say the new property is cheaper, more expensive, higher-priced, or lower-priced
   unless the new property description explicitly includes a numeric price.
4. If the new property description does not include price, say that price comparison
   is not available.
5. When you mention a retrieved listing fact, cite the listing ID in parentheses,
   e.g. (listing_07).
6. If there is not enough information to compare a field, say so clearly.
7. Do not recommend a price or legal action.
"""

_HUMAN = """\
New property description:
{description}

Retrieved similar listings:
{context}

Write the insight now.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def _format_context(listings: list[dict]) -> str:
    parts = []
    for i, listing in enumerate(listings, 1):
        parts.append(
            f"[{i}] ID: {listing['id']}\n"
            f"    Title: {listing['title']}\n"
            f"    Summary: {listing['summary']}\n"
            f"    Similarity score: {listing['score']}"
        )
    return "\n\n".join(parts)


def build_chain():
    llm = get_llm()
    parser = StrOutputParser()

    chain = (
        RunnablePassthrough()
        | (lambda inputs: {
            "description": inputs["description"],
            "context": _format_context(inputs["listings"]),
        })
        | _PROMPT
        | llm
        | parser
    )
    return chain


_chain = None


def get_chain():
    global _chain
    if _chain is None:
        _chain = build_chain()
    return _chain


async def generate_insight(description: str, listings: list[dict]) -> str:
    chain = get_chain()
    result = await chain.ainvoke({"description": description, "listings": listings})
    return result.strip()
