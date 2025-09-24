#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-appuser}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
RAG_STORE_DIR="${RAG_STORE_DIR:-/app/rag_store}"

export AWS_EC2_METADATA_DISABLED="${AWS_EC2_METADATA_DISABLED:-true}"

echo "[rag] preparing volumes..."

mkdir -p /app/.cache /data/vectorstore_faiss "${RAG_STORE_DIR}"

+chown -R "$APP_UID:$APP_GID" /app/.cache /data/vectorstore_faiss "${RAG_STORE_DIR}" 2>/dev/null || true
find /app/.cache -type d -exec chmod u+rwx,g+rwx {} + 2>/dev/null || true
find /data/vectorstore_faiss -type d -exec chmod u+rwx,g+rwx {} + 2>/dev/null || true
find "${RAG_STORE_DIR}" -type d -exec chmod u+rwx,g+rwx {} + 2>/dev/null || true

if [ -n "${S3_BUCKET:-}" ]; then
  # аккуратно соберём префикс и завершающий слеш
  _prefix="${S3_PREFIX:-}"
  if [ -n "$_prefix" ]; then
    SRC_URI="s3://${S3_BUCKET}/${_prefix%/}/"
  else
    SRC_URI="s3://${S3_BUCKET}/"
  fi
  echo "[rag] syncing from ${SRC_URI} -> ${RAG_STORE_DIR}/"
  if ! command -v aws >/dev/null 2>&1; then
    echo "[rag] FATAL: aws cli not found in image"; exit 1
  fi
  # если задан кастомный endpoint (MinIO/YC), используем его
  if [ -n "${AWS_ENDPOINT_URL:-}" ]; then
    aws s3 sync "${SRC_URI}" "${RAG_STORE_DIR}/" --endpoint-url "${AWS_ENDPOINT_URL}"
  else
    aws s3 sync "${SRC_URI}" "${RAG_STORE_DIR}/"
  fi
  chown -R "$APP_UID:$APP_GID" "${RAG_STORE_DIR}" 2>/dev/null || true
  echo "[rag] synced tree:"
  ls -R "${RAG_STORE_DIR}" || true
fi

if [ "${RAG_MODE:-structured}" = "structured" ]; then
  if [ ! -f "${PERS_PATH:-}" ] || [ ! -f "${SYSTEM_PROMPT_PATH:-}" ]; then
    echo "[rag] ERROR: required files not found:"
    echo "  PERS_PATH=$PERS_PATH"
    echo "  SYSTEM_PROMPT_PATH=$SYSTEM_PROMPT_PATH"
    echo "[rag] ls -R ${RAG_STORE_DIR}:"
    ls -R "${RAG_STORE_DIR}" || true
    exit 1
  fi
fi

echo "[rag] starting..."
if command -v gosu >/dev/null 2>&1; then
  exec gosu "$APP_USER":"$APP_USER" uvicorn app:app --host 0.0.0.0 --port 8080
else
  exec su -s /bin/sh -c 'uvicorn app:app --host 0.0.0.0 --port 8080' "$APP_USER"
fi
