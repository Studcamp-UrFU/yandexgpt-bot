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

ENV VSTORE_DIR=/data/vectorstore_faiss \
    S3_ACCESS_KEY="" \
    S3_SECRET_KEY="" \
    S3_BUCKET="" \
    S3_PREFIX="" \
    S3_ENDPOINT_URL="https://storage.yandexcloud.net" \
    S3_REGION="ru-central1" \
    EMB_MODEL="sentence-transformers/distiluse-base-multilingual-cased-v2" \
    CHUNK_SIZE="600" \
    CHUNK_OVERLAP="200" \
    MAX_FILES="2000" \
    MAX_CONTEXT_CHARS="3000" \
    APP_USER="appuser" \
    APP_UID="1000" \
    APP_GID="1000"

# ВАЖНО: не переключаемся на USER здесь — entrypoint должен стартовать root'ом,
# чтобы выполнить chown/chmod на примонтированных томах, а затем запустить uvicorn под appuser
VOLUME ["/data/vectorstore_faiss", "/app/.cache"]
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
