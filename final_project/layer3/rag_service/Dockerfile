FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared schemas
COPY shared /service/shared

# Service source
COPY rag_service/app /service/app
COPY rag_service/data /service/data

# Directory for ChromaDB volume
RUN mkdir -p /service/chroma_db

ENV PYTHONPATH=/service
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]