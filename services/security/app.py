import logging
import os
import re
import unicodedata
import uuid
from typing import List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("security-svc")

# --- конфиг через ENV ---
MAX_INPUT_CHARS     = int(os.getenv("MAX_INPUT_CHARS", "16000"))
NEWLINES_THRESHOLD  = int(os.getenv("NEWLINES_THRESHOLD", "50"))
FENCE_THRESHOLD     = int(os.getenv("FENCE_THRESHOLD", "6"))     # число троек `
BASE64_MIN_RUN      = int(os.getenv("BASE64_MIN_RUN", "120"))    # длина «подозрительной» base64-последовательности
EXTRA_PATTERNS_ENV  = os.getenv("SECURITY_EXTRA_PATTERNS", "")   # дополнительные паттерны через \n

# --- базовые паттерны инъекций (ослабили жадность и ограничили захваты) ---
INJECTION_PATTERNS = [
    r"\byour\s+instructions\b",
    r"\byour\s+prompt\b",
    r"\bsystem\s+prompt\b",
    r"\bsystem\s*[:=]\s*",
    r"\bignore\s+previous\s+instructions?\b",
    r"\bdisregard\s+all\s+prior\s+prompts?\b",
    r"\boverride\s+system\s+rules\b",
    r"\bpretend\s+to\s+be\b",
    r"\bfrom\s+now\s+on\b",
    r"\breset\s+your\s+identity\b",
    r"\bnew\s+instructions?\b.*?\b(from|given|are)\b",

    # русские формулировки
    r"\bне\s+следуй\s+предыдущим\s+инструкциям\b",
    r"\bзабудь\s+все\s+инструкции\b",
    r"\bты\s+должен\b.*?\b(игнорировать|забыть|сменить)\b",
    r"\bне\s+говори\b.*?\b(это|что|никому)\b",
    r"\bраскрой\s+секрет\b",
    r"\bвыведи\s+весь\s+промпт\b",

    # запрос на раскрытие prompt’а
    r"\bshow\s+me\s+the\s+system\s+prompt\b",

    # смена роли — ограничиваем захват до конца строки/120 символов
    r"\byou\s+are\s+(?:an?\s+|the\s+)?[^\n]{1,120}\b(assistant|ai|bot|llm|model|hacker|friend|god|master)\b",
    r"\bas\s+a\s+(?:friend|developer|admin|god|expert|hacker)\b",

    # act as — без жадного .*
    r"\bact\s+as\s+(?:if\s+you\s+are\s+|a\s+)?[^\n]{1,120}",
]

# подключаем доп. паттерны из ENV (по строкам)
if EXTRA_PATTERNS_ENV.strip():
    INJECTION_PATTERNS.extend([p for p in EXTRA_PATTERNS_ENV.splitlines() if p.strip()])

COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in INJECTION_PATTERNS]

# эвристики
RE_ZW = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\ufeff]")   # нулевая ширина/bi-di
RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")                    # control, кроме \t \n \r
RE_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{%d,}" % BASE64_MIN_RUN)

def _normalize(t: str) -> str:
    """Unicode NFKC, вырезание zero-width/контрольных, схлопывание пробелов, lower()."""
    t = unicodedata.normalize("NFKC", t or "")
    t = RE_ZW.sub("", t)
    t = RE_CTRL.sub("", t)
    # не убираем переводы строк: они нужны для эвристик
    t = " ".join(t.split()) if t.count("\n") == 0 else t
    return t.lower()

def _suspicious(t: str) -> bool:
    """Быстрые эвристики до regex: длина, много ``` , base64-куски, лавина переводов строк."""
    if len(t) > MAX_INPUT_CHARS:
        return True
    if t.count("\n") > NEWLINES_THRESHOLD:
        return True
    if t.count("```") >= FENCE_THRESHOLD:
        return True
    if RE_BASE64_RUN.search(t):
        return True
    return False

def detect_injection(text: str) -> bool:
    t = _normalize(text)
    if _suspicious(t):
        return True
    return any(p.search(t) for p in COMPILED)

# --- API-модели ---
class DetectReq(BaseModel):
    text: str = Field(..., min_length=1)

class DetectResp(BaseModel):
    is_injection: bool

# --- FastAPI ---
app = FastAPI(title="security-svc", version="1.1.0")

@app.post("/detect", response_model=DetectResp)
def detect(req: DetectReq, request: Request):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    result = detect_injection(req.text)
    log.info("[rid=%s] detect -> %s", rid, result)
    return DetectResp(is_injection=result)

@app.get("/health")
def health():
    return {
        "ok": True,
        "limits": {
            "max_input_chars": MAX_INPUT_CHARS,
            "newlines_threshold": NEWLINES_THRESHOLD,
            "fence_threshold": FENCE_THRESHOLD,
            "base64_min_run": BASE64_MIN_RUN,
        },
        "patterns": len(INJECTION_PATTERNS),
    }
