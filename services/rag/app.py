import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

VSTORE_DIR = Path(os.getenv("VSTORE_DIR", "/data/vectorstore_faiss"))
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")

import boto3
from botocore.config import Config
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

TMP_DIR = Path("/tmp/rag_s3_tmp")
EMB_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"

def _s3_client():
    if not (S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        raise RuntimeError("S3 creds/bucket not set")
    cfg = Config(signature_version="s3v4", s3={"addressing_style":"virtual"})
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY,
        region_name="ru-central1", config=cfg
    )

def _iter_s3_keys(client):
    cont = None
    exts = (".pdf",".txt",".md")
    while True:
        kwargs = {"Bucket": S3_BUCKET, "Prefix": S3_PREFIX} if S3_PREFIX else {"Bucket": S3_BUCKET}
        if cont: kwargs["ContinuationToken"] = cont
        resp = client.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.lower().endswith(exts):
                yield k
        if resp.get("IsTruncated"): cont = resp.get("NextContinuationToken")
        else: break

def _load_local(path: Path):
    if path.suffix.lower()==".pdf": return PyPDFLoader(str(path)).load()
    if path.suffix.lower() in {".txt",".md"}: return TextLoader(str(path), encoding="utf-8").load()
    return []

def load_corpus_s3() -> list:
    c = _s3_client()
    TMP_DIR.mkdir(exist_ok=True, parents=True)
    docs, any_found = [], False
    for key in _iter_s3_keys(c):
        any_found = True
        local = TMP_DIR / key.replace("/", "__")
        local.parent.mkdir(parents=True, exist_ok=True)
        c.download_file(S3_BUCKET, key, str(local))
        docs.extend(_load_local(local))
    if not any_found:
        raise RuntimeError("No *.pdf|*.txt|*.md files in S3 bucket/prefix")
    return [d for d in docs if getattr(d, "page_content", "").strip()]

def build_store(docs: list, chunk_size=600, chunk_overlap=200):
    if not docs: raise RuntimeError("Empty corpus")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n","\n"," ",""]
    )
    chunks = splitter.split_documents(docs)
    if not chunks: raise RuntimeError("Cannot split corpus")
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vs = FAISS.from_documents(chunks, emb)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VSTORE_DIR))
    return vs

def load_store():
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return FAISS.load_local(str(VSTORE_DIR), emb, allow_dangerous_deserialization=True)

# --- fastapi ---
app = FastAPI(title="rag-svc")
_vs = None  # lazy

def _ensure_vs():
    global _vs
    if _vs is None:
        _vs = load_store()
    return _vs

class CtxReq(BaseModel):
    query: str
    k: int = 4
    max_chars: int = 3000

class CtxResp(BaseModel):
    context: str

class RetRespItem(BaseModel):
    source: str | None = None
    page: int | None = None
    text: str

@app.post("/context", response_model=CtxResp)
def context(req: CtxReq):
    vs = _ensure_vs()
    docs = vs.as_retriever(search_kwargs={"k": req.k}).invoke(req.query)
    text = "\n\n".join(d.page_content for d in docs)[:req.max_chars]
    return CtxResp(context=text)

@app.get("/retrieve", response_model=list[RetRespItem])
def retrieve(query: str, k: int = 4):
    vs = _ensure_vs()
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(query)
    out = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        out.append(RetRespItem(
            source=md.get("source"), page=md.get("page"), text=d.page_content
        ))
    return out

@app.post("/reindex")
def reindex():
    global _vs
    docs = load_corpus_s3()
    _vs = build_store(docs)
    return {"ok": True}

@app.get("/health")
def health(): return {"ok": True}
