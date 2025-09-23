#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-appuser}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

echo "[rag] preparing volumes..."

mkdir -p /app/.cache /data/vectorstore_faiss

chown -R "$APP_UID:$APP_GID" /app/.cache /data/vectorstore_faiss 2>/dev/null || true
find /app/.cache -type d -exec chmod u+rwx,g+rwx {} + 2>/dev/null || true
find /data/vectorstore_faiss -type d -exec chmod u+rwx,g+rwx {} + 2>/dev/null || true

echo "[rag] starting..."
if command -v gosu >/dev/null 2>&1; then
  exec gosu "$APP_USER":"$APP_USER" uvicorn app:app --host 0.0.0.0 --port 8080
else
  exec su -s /bin/sh -c 'uvicorn app:app --host 0.0.0.0 --port 8080' "$APP_USER"
fi