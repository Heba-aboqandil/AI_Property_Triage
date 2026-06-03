FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY image_analyser/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared schemas
COPY shared /service/shared

# Service source
COPY image_analyser/app /service/app

RUN mkdir -p /service/checkpoints

ENV PYTHONPATH=/service
ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
