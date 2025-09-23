import os
import time
import logging
import uuid
import json

import jwt
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from requests import RequestException, Timeout
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------- логирование ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

# ---------- конфиг через ENV с дефолтами ----------
SECURITY_URL   = os.getenv("SECURITY_URL",   "http://security-svc:8080")
MODERATION_URL = os.getenv("MODERATION_URL", "http://moderation-svc:8080")
RAG_URL        = os.getenv("RAG_URL",        "http://rag-svc:8080")

FOLDER_ID           = os.getenv("FOLDER_ID")
SERVICE_ACCOUNT_ID  = os.getenv("SERVICE_ACCOUNT_ID")
KEY_ID              = os.getenv("KEY_ID")
PRIVATE_KEY         = os.getenv("PRIVATE_KEY")

# LLM настройки (переключаемые без правки кода)
MODEL_URI   = os.getenv("MODEL_URI",   "")  # если пусто — соберём ниже из FOLDER_ID
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS",  "1000"))

# Ограничения ввода
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "8000"))

# Yandex Cloud endpoints
IAM_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Кэш IAM-токена
_IAM_TOKEN: str | None = None
_IAM_EXP: int = 0

# HTTP сессия с ретраями
_session = requests.Session()
_retry = Retry(
    total=2,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST", "HEAD"])
)
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))

# ---------- модели ----------
class AskReq(BaseModel):
    question: str = Field(min_length=1)

class AskResp(BaseModel):
    answer: str

# ---------- приложение ----------
app = FastAPI(title="api-gateway", version="1.0.0")


def _env_ok() -> bool:
    """Проверяет, что все критичные переменные окружения заданы."""
    return all([FOLDER_ID, SERVICE_ACCOUNT_ID, KEY_ID, PRIVATE_KEY])


def _get_iam_token() -> str:
    """Получить/рефрешнуть IAM token для работы с API Yandex.Cloud (с кэшем)."""
    global _IAM_TOKEN, _IAM_EXP
    now = int(time.time())

    if _IAM_TOKEN and now < (_IAM_EXP - 60):
        return _IAM_TOKEN

    if not _env_ok():
        raise HTTPException(500, "Missing FOLDER_ID/SERVICE_ACCOUNT_ID/KEY_ID/PRIVATE_KEY")

    pk = (PRIVATE_KEY or "").replace("\\n", "\n")
    payload = {
        "aud": IAM_URL,
        "iss": SERVICE_ACCOUNT_ID,
        "iat": now,
        "exp": now + 3600,
    }

    try:
        jws = jwt.encode(payload, pk, algorithm="PS256", headers={"kid": KEY_ID})
    except Exception as e:
        log.exception("JWT signing failed")
        raise HTTPException(500, f"jwt signing error: {e}")

    try:
        r = _session.post(IAM_URL, json={"jwt": jws}, timeout=10)
        r.raise_for_status()
        data = r.json()
        token = data.get("iamToken")
        if not token:
            raise KeyError("iamToken missing")
        _IAM_TOKEN = token
        _IAM_EXP = now + 3500  # чуть меньше часа, чтобы освежить заранее
        return _IAM_TOKEN
    except (RequestException, ValueError, KeyError) as e:
        body = getattr(r, "text", "")
        log.error("IAM error: %s | body=%s", e, body[:500])
        raise HTTPException(502, f"IAM error: {e}")


def _security_blocked(text: str, request_id: str) -> bool:
    """Проверка ввода пользователя через security-svc (fail-closed)."""
    try:
        r = _session.post(
            f"{SECURITY_URL}/detect",
            json={"text": text},
            headers={"X-Request-ID": request_id},
            timeout=5,
        )
        r.raise_for_status()
        return bool(r.json().get("is_injection", False))
    except (Timeout, RequestException, ValueError) as e:
        log.warning("[rid=%s] security-svc error: %s", request_id, e)
        return True  # fail-closed


def _moderation_blocked(text: str, request_id: str) -> bool:
    """Проверка ввода пользователя через moderation-svc (fail-closed)."""
    try:
        r = _session.post(
            f"{MODERATION_URL}/moderate",
            json={"text": text},
            headers={"X-Request-ID": request_id},
            timeout=10,
        )
        r.raise_for_status()
        return bool(r.json().get("malicious", False))
    except (Timeout, RequestException, ValueError) as e:
        log.warning("[rid=%s] moderation-svc error: %s", request_id, e)
        return True  # fail-closed


def _rag_context(question: str, request_id: str, k: int = 4, max_chars: int = 2500) -> str:
    """Получает контекст из rag-svc для заданного вопроса."""
    try:
        r = _session.post(
            f"{RAG_URL}/context",
            json={"query": question, "k": k, "max_chars": max_chars},
            headers={"X-Request-ID": request_id},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("context", "") or ""
    except (RequestException, ValueError) as e:
        log.warning("[rid=%s] rag-svc error: %s", request_id, e)
        return ""  # не валим ответ — просто без контекста


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq, request: Request):
    request_id = str(uuid.uuid4())
    q = (req.question or "").strip()

    if not q:
        raise HTTPException(400, "empty question")

    # обрежем чрезмерно длинный ввод
    if len(q) > MAX_QUESTION_CHARS:
        log.info("[rid=%s] question trimmed from %d to %d", request_id, len(q), MAX_QUESTION_CHARS)
        q = q[:MAX_QUESTION_CHARS]

    # security/moderation — fail-closed
    if _security_blocked(q, request_id) or _moderation_blocked(q, request_id):
        return AskResp(answer="Ваш запрос не может быть обработан, так как нарушает правила использования.")

    # контекст RAG (по возможности)
    ctx = _rag_context(q, request_id)

    system_prompt = (
        "Ты — корпоративный ассистент. Отвечай строго на основе предоставленных документов. "
        "Если информации по вопросу в документах совсем нет, дай короткий ответ: 'В документах не указано'. "
        "Если часть ответа можно составить по документам — отвечай только этой частью, без добавления 'В документах не указано'. "
        "Не придумывай фактов вне контекста."
        f"\n\nКонтекст из документов:\n{ctx}"
    )

    # подготовка запроса к LLM
    token = _get_iam_token()
    model_uri = MODEL_URI or (f"gpt://{FOLDER_ID}/yandexgpt-lite")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-folder-id": FOLDER_ID,
        "X-Request-ID": request_id,
    }
    payload = {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": TEMPERATURE, "maxTokens": MAX_TOKENS},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": q},
        ],
    }

    try:
        r = _session.post(LLM_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        jr = r.json()
        # безопасный парсинг
        alt = ((jr or {}).get("result") or {}).get("alternatives")
        if not alt or not isinstance(alt, list):
            raise KeyError("alternatives missing")
        msg = (alt[0] or {}).get("message") or {}
        ans = msg.get("text")
        if not isinstance(ans, str) or not ans.strip():
            raise KeyError("empty text")
        return AskResp(answer=ans)
    except (RequestException, ValueError, KeyError) as e:
        body = getattr(r, "text", "")
        log.error("[rid=%s] LLM error: %s | body=%s", request_id, e, body[:500])
        raise HTTPException(502, f"LLM error: {e}")


@app.get("/health")
def health():
    """Проверка жизнеспособности сервиса и зависимостей (best-effort)."""
    deps = {}
    for name, url in {
        "security": f"{SECURITY_URL}/health",
        "moderation": f"{MODERATION_URL}/health",
        "rag": f"{RAG_URL}/health",
    }.items():
        try:
            r = _session.get(url, timeout=2)
            deps[name] = (r.status_code == 200)
        except RequestException:
            deps[name] = False

    return {"ok": True, "env_ok": _env_ok(), "deps": deps}
