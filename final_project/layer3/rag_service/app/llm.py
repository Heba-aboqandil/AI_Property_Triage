"""
Llama.cpp integration via langchain-community LlamaCpp wrapper.

The GGUF model file is expected at the path set by the MODEL_PATH env var.
Default: /service/models/llama3.gguf  (mounted from EBS on EC2).

On a resource-constrained t3.large (no GPU), we keep n_ctx small and use
all available CPU threads.
"""

import os
from functools import lru_cache

from langchain_community.llms import LlamaCpp

MODEL_PATH = os.getenv("MODEL_PATH", "/service/models/llama3.gguf")
N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))
N_THREADS = int(os.getenv("LLAMA_N_THREADS", "4"))
MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.1"))


@lru_cache(maxsize=1)
def get_llm() -> LlamaCpp:
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
