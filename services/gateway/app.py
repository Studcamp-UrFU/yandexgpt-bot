import os, time, requests, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("gateway"); logging.basicConfig(level=logging.INFO)

SECURITY_URL = os.getenv("SECURITY_URL", "http://security-svc:8080")
MODERATION_URL = os.getenv("MODERATION_URL", "http://moderation-svc:8080")
RAG_URL = os.getenv("RAG_URL", "http://rag-svc:8080")

FOLDER_ID = os.getenv("FOLDER_ID")
SERVICE_ACCOUNT_ID = os.getenv("SERVICE_ACCOUNT_ID")
KEY_ID = os.getenv("KEY_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
IAM_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

IAM_TOKEN, IAM_EXP = None, 0
def get_iam_token():
    global IAM_TOKEN, IAM_EXP
    now = int(time.time())
    if IAM_TOKEN and now < IAM_EXP:
        return IAM_TOKEN
    import jwt
    payload = {'aud': IAM_URL, 'iss': SERVICE_ACCOUNT_ID, 'iat': now, 'exp': now + 360}
    pk = PRIVATE_KEY.replace("\\n", "\n") if PRIVATE_KEY else PRIVATE_KEY
    encoded = jwt.encode(payload, pk, algorithm="PS256", headers={'kid': KEY_ID})
    r = requests.post(IAM_URL, json={'jwt': encoded}, timeout=10)
    r.raise_for_status()
    IAM_TOKEN = r.json()['iamToken']; IAM_EXP = now + 3500
    return IAM_TOKEN

class AskReq(BaseModel):
    question: str

class AskResp(BaseModel):
    answer: str

app = FastAPI(title="api-gateway")

def _security_blocked(text: str) -> bool:
    r = requests.post(f"{SECURITY_URL}/detect", json={"text": text}, timeout=5)
    r.raise_for_status()
    return r.json()["is_injection"]

def _moderation_blocked(text: str) -> bool:
    r = requests.post(f"{MODERATION_URL}/moderate", json={"text": text}, timeout=15)
    r.raise_for_status()
    return r.json()["malicious"]

def _rag_context(question: str, k=4, max_chars=2500) -> str:
    r = requests.post(f"{RAG_URL}/context", json={"query": question, "k": k, "max_chars": max_chars}, timeout=20)
    r.raise_for_status()
    return r.json()["context"]

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "empty question")

    if _security_blocked(q):
        return AskResp(answer="ban")

    if _moderation_blocked(q):
        return AskResp(answer="ban2")

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
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {get_iam_token()}',
        'x-folder-id': FOLDER_ID
    }
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": 1000},
        "messages": [{"role":"system","text":system_prompt}, {"role":"user","text":q}]
    }
    r = requests.post(LLM_URL, headers=headers, json=data, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error: {r.text}")
    ans = r.json()['result']['alternatives'][0]['message']['text']
    return AskResp(answer=ans)

@app.get("/health")
def health(): return {"ok": True}
