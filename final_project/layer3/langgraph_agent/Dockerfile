FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY langgraph_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared schemas
COPY shared /service/shared

# Service source
COPY langgraph_agent/app /service/app

ENV PYTHONPATH=/service
ENV PYTHONUNBUFFERED=1

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
