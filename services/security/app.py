import logging
import os
import re
import unicodedata
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("security-svc")

MAX_INPUT_CHARS = 16000
NEWLINES_THRESHOLD = 50
FENCE_THRESHOLD = 6
BASE64_MIN_RUN = 120
EXTRA_PATTERNS_ENV = ""

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

    r"\bне\s+следуй\s+предыдущим\s+инструкциям\b",
    r"\bзабудь\s+все\s+инструкции\b",
    r"\bты\s+должен\b.*?\b(игнорировать|забыть|сменить)\b",
    r"\bне\s+говори\b.*?\b(это|что|никому)\b",
    r"\bраскрой\s+секрет\b",
    r"\bвыведи\s+весь\s+промпт\b",

    r"\bshow\s+me\s+the\s+system\s+prompt\b",

    r"\byou\s+are\s+(?:an?\s+|the\s+)?[^\n]{1,120}\b(assistant|ai|bot|llm|model|hacker|friend|god|master)\b",
    r"\bas\s+a\s+(?:friend|developer|admin|god|expert|hacker)\b",

    r"\bact\s+as\s+(?:if\s+you\s+are\s+|a\s+)?[^\n]{1,120}",
]

COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in INJECTION_PATTERNS]

RE_ZW = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\ufeff]")
RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
RE_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{%d,}" % BASE64_MIN_RUN)

def _normalize(t: str) -> str:
    """Unicode NFKC, вырезание zero-width/контрольных, схлопывание пробелов, lower()."""
    t = unicodedata.normalize("NFKC", t or "")
    t = RE_ZW.sub("", t)
    t = RE_CTRL.sub("", t)
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


class DetectReq(BaseModel):
    text: str = Field(..., min_length=1)


class DetectResp(BaseModel):
    is_injection: bool


app = FastAPI(title="security-svc")

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
