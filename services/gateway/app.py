import os
import time
import threading
import logging
import uuid
from datetime import datetime, timedelta

import jwt
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from requests import RequestException, Timeout
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, text
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

SECURITY_URL = "http://security-svc:8080"
MODERATION_URL = "http://moderation-svc:8080"
RAG_URL = "http://rag-svc:8080"

FOLDER_ID = os.getenv("FOLDER_ID")
SERVICE_ACCOUNT_ID = os.getenv("SERVICE_ACCOUNT_ID")
KEY_ID = os.getenv("KEY_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

MODEL_URI = ""
TEMPERATURE = 0.6
MAX_TOKENS = 1000

MAX_QUESTION_CHARS = 8000

IAM_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

_IAM_TOKEN: str | None = None
_IAM_EXP: int = 0
_IAM_LOCK = threading.Lock()

_session = requests.Session()
_retry = Retry(
    total = 2,
    backoff_factor = 0.3,
    status_forcelist = (429, 500, 502, 503, 504),
    allowed_methods = frozenset(["GET", "POST", "HEAD"])
)
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))

HISTORY_DB_URL = "sqlite:////data/history.db"
HISTORY_MAX_TURNS = 8
HISTORY_MAX_DAYS = 7

engine = create_engine(HISTORY_DB_URL, future=True)

def _init_db():
    try:
        os.makedirs("/data", exist_ok=True)
    except Exception:
        pass
    with engine.begin() as conn:
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            chat_id TEXT,
            role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
            content TEXT NOT NULL,
            ts TIMESTAMP NOT NULL
        )
        """)
        conn.exec_driver_sql("""
        CREATE INDEX IF NOT EXISTS idx_hist_user_ts
        ON chat_history(user_id, ts)
        """)


def save_message(user_id, chat_id, role, content):
    now = datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_history(user_id, chat_id, role, content, ts) VALUES (:u,:c,:r,:t,:ts)"),
            {"u": user_id, "c": chat_id or "", "r": role, "t": content, "ts": now},
        )
        cutoff = now - timedelta(days=HISTORY_MAX_DAYS)
        conn.execute(text("DELETE FROM chat_history WHERE ts < :cutoff"), {"cutoff": cutoff})

def fetch_last_turns(user_id, limit_msgs):
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT role, content, ts FROM chat_history WHERE user_id = :u ORDER BY ts DESC LIMIT :lim"),
            {"u": user_id, "lim": int(limit_msgs)},
        ).mappings().all()
    return list(reversed(rows))


class AskReq(BaseModel):
    question: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    chat_id: str | None = None

class AskResp(BaseModel):
    answer: str


app = FastAPI(title="api-gateway")


def _env_ok() -> bool:
    """Проверяет, что все переменные окружения заданы."""
    return all([FOLDER_ID, SERVICE_ACCOUNT_ID, KEY_ID, PRIVATE_KEY])


def _get_iam_token() -> str:
    """Получить IAM token для работы с API Yandex.Cloud."""
    global _IAM_TOKEN, _IAM_EXP
    now = int(time.time())

    with _IAM_LOCK:
        if _IAM_TOKEN and now < (_IAM_EXP - 60):
            return _IAM_TOKEN

    if not _env_ok():
        raise HTTPException(500, "Missing FOLDER_ID/SERVICE_ACCOUNT_ID/KEY_ID/PRIVATE_KEY")

    pk = PRIVATE_KEY.replace("\\n", "\n")
    payload = {
        "aud": IAM_URL,  # для кого токен
        "iss": SERVICE_ACCOUNT_ID,  # кто выпускает токен
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
        _IAM_EXP = now + 3500
        return _IAM_TOKEN
    except (RequestException, ValueError, KeyError) as e:
        body = getattr(r, "text", "")
        log.error("IAM error: %s | body=%s", e, body[:500])
        raise HTTPException(502, f"IAM error: {e}")


def _security_blocked(text: str, request_id: str) -> bool:
    """Проверка ввода пользователя через security-svc."""
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
        return True


def _moderation_blocked(text: str, request_id: str) -> bool:
    """Проверка ввода пользователя через moderation-svc."""
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
        return True


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
        return ""


@app.post("/ask", response_model=AskResp)
def ask(req: AskReq, request: Request):
    request_id = str(uuid.uuid4())
    q = (req.question or "").strip()

    if not q:
        raise HTTPException(400, "empty question")

    if len(q) > MAX_QUESTION_CHARS:
        log.info("[rid=%s] question trimmed from %d to %d", request_id, len(q), MAX_QUESTION_CHARS)
        q = q[:MAX_QUESTION_CHARS]

    save_message(req.user_id, req.chat_id, "user", q)

    if _security_blocked(q, request_id) or _moderation_blocked(q, request_id):
        blocked_ans = "Ваш запрос не может быть обработан, так как нарушает правила использования."
        save_message(req.user_id, req.chat_id, "assistant", blocked_ans)
        return AskResp(answer=blocked_ans)

    ctx = _rag_context(q, request_id)

    turns = fetch_last_turns(req.user_id, HISTORY_MAX_TURNS * 2)

    base_system = (
        "Ты — корпоративный ассистент, который работает строго по двум источникам:\n"
        "1) Документы (даются в контексте).\n"
        "2) История диалога с пользователем (несколько последних сообщений).\n\n"
        "Твои главные принципы работы:\n"
        "- Факты и знания о внешнем мире берёшь только из документов.\n"
        "- Историю диалога используешь для ответов о самом диалоге (что спрашивали/отвечали ранее).\n"
        "- Если вопрос комбинированный (часть про факты, часть про диалог) — отвечай на обе части отдельно.\n"
        "- Никогда не выдумывай: если в документах нет сведений — скажи об этом своими словами.\n\n"
        "Правила работы с документами:\n"
        "- Если ответ есть — используй его (перескажи своими словами или процитируй кратко).\n"
        "- Если сведений нет — вежливо сообщи об отсутствии. Формулировки можно варьировать:\n"
        "  • «в предоставленных материалах я не нашёл информации по этому вопросу»\n"
        "  • «документы не содержат сведений об этом»\n"
        "  • «к сожалению, в документах про это ничего не сказано»\n"
        "  • «в материалах отсутствуют данные по этой теме»\n\n"
        "Правила работы с историей диалога:\n"
        "- На вопросы о самом диалоге опирайся на историю и отвечай честно, без искажения смысла.\n"
        "- Если история пуста — сообщи об этом и предложи начать сначала.\n\n"
        "Комбинированные запросы:\n"
        "- Дай раздельный ответ: фактическую часть — по документам (или скажи, что сведений нет),\n"
        "  диалоговую часть — по истории.\n\n"
        "- Никогда не замещай одну часть другой, отвечай на обе.\n\n"
        "Тон общения:\n"    
        "- Будь вежливым, дружелюбным, внимательным.\n"
        "- Пиши так, чтобы тебя легко было читать: делай короткие абзацы, списки, пояснения.\n"
        "- Если информации нет — не оставляй пустого ответа, а мягко объясни, что в документах ничего не сказано.\n\n"
        "Чего делать нельзя:\n"
        "- Не придумывай факты, которых нет в документах.\n"
        "- Не выдавай догадки или фантазии.\n"
        "- Не используй личное мнение, только документы и историю диалога.\n"
        "- Не игнорируй вопросы про историю: если просят вспомнить, обязательно смотри в диалог.\n"
    )

    messages = [{"role": "system", "text": base_system}]
    if ctx:
        messages.append({"role": "system", "text": f"Контекст из документов:\n{ctx}"})
    for t in turns[-(HISTORY_MAX_TURNS * 2):]:
        role = "user" if t["role"] == "user" else "assistant"
        messages.append({"role": role, "text": t["content"]})
    messages.append({"role": "user", "text": q})

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
        "messages": messages,  
    }

    try:
        r = _session.post(LLM_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        jr = r.json()
        alt = ((jr or {}).get("result") or {}).get("alternatives")
        if not alt or not isinstance(alt, list):
            raise KeyError("alternatives missing")
        msg = (alt[0] or {}).get("message") or {}
        ans = msg.get("text")
        if not isinstance(ans, str) or not ans.strip():
            raise KeyError("empty text")
        save_message(req.user_id, req.chat_id, "assistant", ans)
        return AskResp(answer=ans)
    except (RequestException, ValueError, KeyError) as e:
        body = getattr(r, "text", "")
        log.error("[rid=%s] LLM error: %s | body=%s", request_id, e, body[:500])
        raise HTTPException(502, f"LLM error: {e}")


@app.get("/history")
def get_history(user_id: str, limit: int = 20):
    limit = max(1, min(int(limit), 100))
    rows = fetch_last_turns(user_id, limit)
    items = []
    for rrow in rows:
        items.append({
            "role": rrow["role"],
            "content": rrow["content"],
            "ts": rrow["ts"].isoformat() if hasattr(rrow["ts"], "isoformat") else str(rrow["ts"])
        })
    return {"items": items}

@app.delete("/history")
def delete_history(user_id: str):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM chat_history WHERE user_id = :u"), {"u": user_id})
    return {"ok": True}

@app.get("/health")
def health():
    """Проверка жизнеспособности сервиса."""
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

_init_db()
