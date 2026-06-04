import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.1"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
