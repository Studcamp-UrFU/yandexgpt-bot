FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# для faiss/torch иногда нужен gcc+glibc-dev при сборке — на slim обычно не нужно,
# но оставлю минимальные системные утилиты (можно удалить, если не понадобится)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) зависимости (тяжёлые — кэшируй по максимуму)
COPY services/rag/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) код и entrypoint
COPY services/rag/app.py /app/app.py
COPY services/rag/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 3) параметры rag
ENV VSTORE_DIR=/data/vectorstore_faiss \
    REINDEX_DELAY=0 \
    S3_ACCESS_KEY="" \
    S3_SECRET_KEY="" \
    S3_BUCKET="" \
    S3_PREFIX="" \
    S3_ENDPOINT_URL="https://storage.yandexcloud.net"

# 4) выделяем том под индекс
VOLUME ["/data/vectorstore_faiss"]
EXPOSE 8080

# 5) авто-реиндекс только в этом контейнере
ENTRYPOINT ["/app/entrypoint.sh"]
