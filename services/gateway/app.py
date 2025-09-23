import os
import time
import logging
from typing import Optional

import requests
import jwt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

SECURITY_URL = os.getenv("SECURITY_URL", "http://security-svc:8080")
MODERATION_URL = os.getenv("MODERATION_URL", "http://moderation-svc:8080")
RAG_URL = os.getenv("RAG_URL", "http://rag-svc:8080")

FOLDER_ID = os.getenv("FOLDER_ID")
SERVICE_ACCOUNT_ID = os.getenv("SERVICE_ACCOUNT_ID")
KEY_ID = os.getenv("KEY_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

IAM_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

_IAM_TOKEN: Optional[str] = None
_IAM_EXP: int = 0  

class AskReq(BaseModel):
    question: str


class AskResp(BaseModel):
    answer: str


app = FastAPI(title="api-gateway", version="1.0.0")


def _env_ok() -> bool:
    return all([FOLDER_ID, SERVICE_ACCOUNT_ID, KEY_ID, PRIVATE_KEY])


def _get_iam_token() -> str:
    """Получить IAM token по PS256. Требует PyJWT[crypto]."""
    global _IAM_TOKEN, _IAM_EXP
    now = int(time.time())
    if _IAM_TOKEN and now < (_IAM_EXP - 60):
        return _IAM_TOKEN

    if not _env_ok():
        raise HTTPException(500, "Missing FOLDER_ID/SERVICE_ACCOUNT_ID/KEY_ID/PRIVATE_KEY")

    pk = PRIVATE_KEY.replace("\\n", "\n") if PRIVATE_KEY else PRIVATE_KEY
    payload = {"aud": IAM_URL, "iss": SERVICE_ACCOUNT_ID, "iat": now, "exp": now + 3600}

    try:
        jws = jwt.encode(payload, pk, algorithm="PS256", headers={"kid": KEY_ID})
    except Exception as e:
        log.exception("JWT signing failed")
        raise HTTPException(500, f"jwt signing error: {e}")

    try:
        r = requests.post(IAM_URL, json={"jwt": jws}, timeout=10)
        r.raise_for_status()
        data = r.json()
        _IAM_TOKEN = data["iamToken"]
        _IAM_EXP = now + 3500  
        return _IAM_TOKEN
    except requests.RequestException as e:
        log.exception("IAM request failed")
        raise HTTPException(502, f"IAM error: {e}")
    except KeyError:
        log.error("Unexpected IAM response: %s", r.text if 'r' in locals() else "")
        raise HTTPException(502, "IAM bad response")


def _security_blocked(text: str) -> bool:
    try:
        r = requests.post(f"{SECURITY_URL}/detect", json={"text": text}, timeout=5)
        r.raise_for_status()
        return bool(r.json().get("is_injection", False))
    except Exception:
        return False


def _moderation_blocked(text: str) -> bool:
    try:
        r = requests.post(f"{MODERATION_URL}/moderate", json={"text": text}, timeout=10)
        r.raise_for_status()
        return bool(r.json().get("malicious", False))
    except Exception:
        return False


def _rag_context(question: str, k: int = 4, max_chars: int = 2500) -> str:
    r = requests.post(
        f"{RAG_URL}/context",
        json={"query": question, "k": k, "max_chars": max_chars},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("context", "")


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "empty question")

    if _security_blocked(q):
        return AskResp(answer="⚠️ Запрос похож на промпт-инъекцию.")
    if _moderation_blocked(q):
        return AskResp(answer="⚠️ Запрос нарушает правила (модерация).")

    ctx = ""
    try:
        ctx = _rag_context(q)
    except Exception:
        ctx = ""

    system_prompt = (
        "Ты — корпоративный ассистент. Отвечай строго по документам. "
        "Если информации нет — скажи 'В документах не указано'.\n\n"
        f"Контекст из документов:\n{ctx}"
    )

    token = _get_iam_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-folder-id": FOLDER_ID, 
    }
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": 1000},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": q},
        ],
    }

    try:
        r = requests.post(LLM_URL, headers=headers, json=data, timeout=30)
        if r.status_code != 200:
            raise HTTPException(502, f"LLM error: {r.text}")
        ans = r.json()["result"]["alternatives"][0]["message"]["text"]
        return AskResp(answer=ans)
    except requests.RequestException as e:
        raise HTTPException(502, f"LLM network error: {e}")


@app.get("/health")
def health():
    return {"ok": True, "env_ok": _env_ok()}
