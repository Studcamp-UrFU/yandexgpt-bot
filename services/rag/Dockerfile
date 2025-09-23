FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Кеши моделей (чтобы не лез в /.cache и не ругался на права)
ENV HF_HOME=/app/.cache \
    HUGGINGFACE_HUB_CACHE=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

COPY services/rag/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-input --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-input -r requirements.txt

COPY services/rag/app.py /app/app.py
COPY services/rag/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV VSTORE_DIR=/data/vectorstore_faiss \
    S3_ACCESS_KEY="" \
    S3_SECRET_KEY="" \
    S3_BUCKET="" \
    S3_PREFIX="" \
    S3_ENDPOINT_URL="https://storage.yandexcloud.net"

VOLUME ["/data/vectorstore_faiss", "/app/.cache"]
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
