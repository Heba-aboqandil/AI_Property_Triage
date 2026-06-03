FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl git \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-python requires cmake to build from source (CPU-only)
ENV CMAKE_ARGS="-DLLAMA_BLAS=OFF"
ENV FORCE_CMAKE=1

COPY rag_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared schemas
COPY shared /service/shared

# Service source
COPY rag_service/app /service/app
COPY rag_service/data /service/data

# Directories for mounted volumes
RUN mkdir -p /service/chroma_db /service/models

ENV PYTHONPATH=/service
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]