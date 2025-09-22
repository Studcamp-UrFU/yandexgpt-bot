from fastapi import FastAPI
from pydantic import BaseModel
import re

INJECTION_PATTERNS = [
    r"\byour instructions\b", r"\byour prompt\b", r"\bsystem prompt\b", r"\bsystem\s*[:=]\s*",
    r"\byou are\b.*?\b(an?|the)\b.*?\b(assistant|ai|bot|llm|model|hacker|friend|god|master)\b",
    r"\bignore\s+previous\s+instructions?\b", r"\bdisregard\s+all\s+prior\s+prompts?\b",
    r"\bas\s+a\s+(friend|developer|admin|god|expert|hacker)\b",
    r"\bact\s+as\s+(if\s+you\s+are|a)\s+(.*)",
    r"\bне\s+следуй\s+предыдущим\s+инструкциям\b", r"\bзабудь\s+все\s+инструкции\b",
    r"\bты\s+должен\b.*?\b(игнорировать|забыть|сменить)\b", r"\boverride\s+system\s+rules\b",
    r"\bpretend\s+to\s+be\b", r"\bfrom\s+now\s+on\b", r"\breset\s+your\s+identity\b",
    r"\bnew\s+instructions?\b.*?\b(from|given|are)\b",
    r"\boutput\s+only\b", r"\bdo\s+not\s+say\b", r"\bне\s+говори\b.*?\b(это|что|никому)\b",
    r"\bsecret\s+word\b", r"\bраскрой\s+секрет\b", r"\bвыведи\s+весь\s+промпт\b",
    r"\bshow\s+me\s+the\s+system\s+prompt\b",
]
COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE | re.DOTALL) for p in INJECTION_PATTERNS]

def _normalize(t: str) -> str:
    return " ".join((t or "").split()).lower()

def detect_injection(text: str) -> bool:
    t = _normalize(text)
    return any(p.search(t) for p in COMPILED)

class DetectReq(BaseModel):
    text: str

class DetectResp(BaseModel):
    is_injection: bool

app = FastAPI(title="security-svc")

@app.post("/detect", response_model=DetectResp)
def detect(req: DetectReq):
    return DetectResp(is_injection=detect_injection(req.text))

@app.get("/health")
def health():
    return {"ok": True}
