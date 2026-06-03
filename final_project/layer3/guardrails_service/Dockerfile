FROM python:3.11-slim

WORKDIR /service

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY guardrails_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY shared /service/shared

# Copy service source
COPY guardrails_service/app /service/app
COPY guardrails_service/rails /service/rails

ENV PYTHONPATH=/service
ENV PYTHONUNBUFFERED=1

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
