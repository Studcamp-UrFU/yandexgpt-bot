import os
import shutil
import threading
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VSTORE_DIR = Path("/data/vectorstore_faiss")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_ENDPOINT_URL = "https://storage.yandexcloud.net"

TMP_DIR = Path("/tmp/rag_s3_tmp")
EMB_MODEL = os.getenv(
    "EMB_MODEL",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
)

app = FastAPI(title="rag-svc", version="1.0.0")

_vs = None
_building = False
_build_lock = threading.Lock()


def _s3_client():
    if not (S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        raise RuntimeError("S3 creds/bucket not set")
    cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"},
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="ru-central1",  
        config=cfg,
    )


def _iter_s3_keys(client):
    cont = None
    exts = (".pdf", ".txt", ".md")
    while True:
        kwargs = {"Bucket": S3_BUCKET, "Prefix": S3_PREFIX} if S3_PREFIX else {"Bucket": S3_BUCKET}
        if cont:
            kwargs["ContinuationToken"] = cont
        resp = client.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.lower().endswith(exts):
                yield k
        if resp.get("IsTruncated"):
            cont = resp.get("NextContinuationToken")
        else:
            break


def _load_local(path: Path):
    suf = path.suffix.lower()
    if suf == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suf in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
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
        raise RuntimeError("No *.pdf|*.txt|*.md in S3 bucket/prefix")
    return [d for d in docs if getattr(d, "page_content", "").strip()]


def _make_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_store(docs: list, chunk_size=600, chunk_overlap=200):
    if not docs:
        raise RuntimeError("Empty corpus")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("Cannot split corpus")
    emb = _make_embeddings()
    vs = FAISS.from_documents(chunks, emb)

    if VSTORE_DIR.exists():
        for p in VSTORE_DIR.glob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vs.save_local(str(VSTORE_DIR))
    return vs


def _index_exists() -> bool:
    return (VSTORE_DIR / "index.faiss").exists() and (VSTORE_DIR / "index.pkl").exists()


def load_store():
    if not _index_exists():
        return None
    emb = _make_embeddings()
    return FAISS.load_local(str(VSTORE_DIR), emb, allow_dangerous_deserialization=True)


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
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


def _reindex_internal():
    global _vs, _building
    with _build_lock:
        if _building:
            return
        _building = True
    try:
        docs = load_corpus_s3()
        _vs = build_store(docs)
        print(f"[rag] reindex done: files={len(docs)} dir={VSTORE_DIR}", flush=True)
    except Exception as e:
        print(f"[rag] reindex failed: {e}", flush=True)
    finally:
        _building = False


@app.on_event("startup")
def _auto_reindex_on_start():
    print("[rag] auto-reindex on startup...", flush=True)
    threading.Thread(target=_reindex_internal, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "has_index": _index_exists(), "building": _building}


@app.post("/reindex")
def reindex():
    threading.Thread(target=_reindex_internal, daemon=True).start()
    return {"ok": True, "started": True}


@app.post("/context", response_model=CtxResp)
def context(req: CtxReq):
    vs = _ensure_vs()
    if vs is None:
        return CtxResp(context="")
    docs = vs.as_retriever(search_kwargs={"k": req.k}).invoke(req.query)
    text = "\n\n".join(d.page_content for d in docs)[: req.max_chars]
    return CtxResp(context=text)


@app.get("/retrieve", response_model=List[RetRespItem])
def retrieve(query: str, k: int = 4):
    vs = _ensure_vs()
    if vs is None:
        return []
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(query)
    out: List[RetRespItem] = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        out.append(
            RetRespItem(source=md.get("source"), page=md.get("page"), text=d.page_content)
        )
    return out
