import os
import shutil
import threading
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-svc")

VSTORE_DIR = Path("/data/vectorstore_faiss")

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "")
S3_ENDPOINT_URL = "https://storage.yandexcloud.net"
S3_REGION = "ru-central1"

TMP_DIR = Path("/tmp/rag_s3_tmp")

EMB_MODEL = os.getenv(
    "EMB_MODEL",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
MAX_FILES = 2000
MAX_CONTEXT_CHARS = 3000

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
        region_name=S3_REGION,
        config=cfg,
    )


def _iter_s3_keys(client):
    cont = None
    exts = (".pdf", ".txt", ".md")
    seen = 0
    while True:
        kwargs = {"Bucket": S3_BUCKET}
        if S3_PREFIX:
            kwargs["Prefix"] = S3_PREFIX
        if cont:
            kwargs["ContinuationToken"] = cont
        resp = client.list_objects_v2(**kwargs)

        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.lower().endswith(exts):
                yield k
                seen += 1
                if seen >= MAX_FILES:
                    log.warning("hit MAX_FILES=%d, stopping listing", MAX_FILES)
                    return

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

    docs = [d for d in docs if getattr(d, "page_content", "").strip()]
    if not docs:
        raise RuntimeError("All documents are empty after load")

    return docs


def _make_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMB_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )


def build_store(docs: list, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not docs:
        raise RuntimeError("Empty corpus")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", " ", ""],
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


def _cleanup_tmp():
    shutil.rmtree(TMP_DIR, ignore_errors=True)


class CtxReq(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = 4
    max_chars: int = MAX_CONTEXT_CHARS


class CtxResp(BaseModel):
    context: str


class RetRespItem(BaseModel):
    source: str | None = None
    page: int | None = None
    text: str


def _reindex_internal(rid: str):
    global _vs, _building
    try:
        docs = load_corpus_s3()
        _vs = build_store(docs)
        log.info("[rid=%s] reindex done: files=%d dir=%s", rid, len(docs), VSTORE_DIR)
    except Exception as e:
        log.error("[rid=%s] reindex failed: %s", rid, e)
    finally:
        _cleanup_tmp()
        with _build_lock:
            _building = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rid = str(uuid.uuid4())
    log.info("[rid=%s] auto-reindex on startup...", rid)

    global _building
    with _build_lock:
        if not _building:
            _building = True
            threading.Thread(target=_reindex_internal, args=(rid,), daemon=True).start()

    # yield — точка, где приложение работает
    yield

    _cleanup_tmp()

app = FastAPI(title="rag-svc", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "has_index": _index_exists(), "building": _building, "model": EMB_MODEL}


@app.post("/reindex")
def reindex():
    rid = str(uuid.uuid4())
    with _build_lock:
        if _building:
            raise HTTPException(409, "already building")
        _building = True
    log.info("[rid=%s] manual reindex requested", rid)
    threading.Thread(target=_reindex_internal, args=(rid,), daemon=True).start()
    return {"ok": True, "started": True, "request_id": rid}


@app.post("/context", response_model=CtxResp)
def context(req: CtxReq):
    vs = _ensure_vs()
    if vs is None:
        return CtxResp(context="")
    docs = vs.as_retriever(search_kwargs={"k": req.k}).invoke(req.query)
    text = "\n\n".join(d.page_content for d in docs)[: req.max_chars]
    return CtxResp(context=text)


@app.get("/retrieve", response_model=list[RetRespItem])
def retrieve(query: str, k: int = 4):
    vs = _ensure_vs()
    if vs is None:
        return []
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(query)
    out: list[RetRespItem] = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        out.append(
            RetRespItem(
                source=md.get("source"),
                page=md.get("page"),
                text=d.page_content,
            )
        )
    return out
