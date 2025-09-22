#!/usr/bin/env bash
set -euo pipefail
echo "[rag] reindex on start..."

python - <<'PY'
import requests, time
from multiprocessing import Process

def run():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)

p = Process(target=run, daemon=True); p.start()
time.sleep(1.0)

for _ in range(30):
    try:
        r = requests.post("http://127.0.0.1:8080/reindex", timeout=600)
        print("[rag] reindex:", r.status_code, r.text)
        break
    except Exception:
        time.sleep(1)
PY

exec uvicorn app:app --host 0.0.0.0 --port 8080
