FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gosu libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/.cache \
    HUGGINGFACE_HUB_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache
RUN mkdir -p /app/.cache

COPY services/rag/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-input --upgrade pip setuptools wheel && \
    pip install --no-input -r requirements.txt

COPY services/rag/app.py /app/app.py
COPY services/rag/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME ["/data/vectorstore_faiss", "/app/.cache"]
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
