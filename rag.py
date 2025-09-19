from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

VSTORE_DIR = Path("./vectorstore_faiss")
EMB_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT_URL = "https://storage.yandexcloud.net"
S3_BUCKET = os.getenv("S3_BUCKET")                  
S3_PREFIX = os.getenv("S3_PREFIX", "")             

# Временное хранилище для объектов S3 до индексации
TMP_DIR = Path(".rag_s3_tmp") 


def _require_s3():
    if not (S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        raise RuntimeError(
            "S3 не настроен: нужны S3_ACCESS_KEY, S3_SECRET_KEY и S3_BUCKET "
            "(S3_ENDPOINT_URL/S3_PREFIX — по желанию)."
        )


def _s3_client():
    """Клиент S3 с подписью v4 и регионом для Yandex Object Storage."""
    _require_s3()
    cfg = Config(
        signature_version="s3v4",
        # virtual - если имя бакета соответствует стандарту DNS, иначе - path
        s3={"addressing_style": "virtual"}  
    )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="ru-central1",
        config=cfg,
    )


def _iter_s3_keys(client) -> Iterable[str]:
    """Итерируем все объекты с нужными расширениями под префиксом."""
    continuation = None
    exts = (".pdf", ".txt", ".md")
    while True:
        kwargs = {"Bucket": S3_BUCKET, "Prefix": S3_PREFIX} if S3_PREFIX else {"Bucket": S3_BUCKET}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = client.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.lower().endswith(exts):
                yield key
        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break


def _load_one_local(path: Path):
    """Парсим один файл (уже скачанный из S3) в список LangChain-документов."""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    if path.suffix.lower() in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return []


def load_corpus_s3() -> List:
    """Скачиваем подходящие объекты из S3 во временную папку и парсим."""
    client = _s3_client()
    TMP_DIR.mkdir(exist_ok=True)

    docs: List = []
    any_found = False

    for key in _iter_s3_keys(client):
        any_found = True
        local_path = TMP_DIR / key.replace("/", "__")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(S3_BUCKET, key, str(local_path))
        docs.extend(_load_one_local(local_path))

    if not any_found:
        raise RuntimeError(
            f"S3: не найдено ни одного файла *.pdf|*.txt|*.md в бакете '{S3_BUCKET}'"
            + (f" с префиксом '{S3_PREFIX}'" if S3_PREFIX else "")
        )

    return [d for d in docs if getattr(d, "page_content", "").strip()]


def build_store(docs: Iterable, chunk_size=500, chunk_overlap=150):
    """Бьём документы на чанки, считаем эмбеддинги и сохраняем FAISS."""
    docs = list(docs)
    if not docs:
        raise RuntimeError("Пустой корпус: индексировать нечего.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("Не удалось разбить документы на чанки.")

    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vs = FAISS.from_documents(chunks, emb)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VSTORE_DIR))
    return vs


def load_store():
    """Грузим сохранённый FAISS-векторстор."""
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return FAISS.load_local(str(VSTORE_DIR), emb, allow_dangerous_deserialization=True)


class RAG:
    """Только S3: индексировать бакет, открывать индекс и доставать контекст."""

    def __init__(self):
        self.vs: Optional[FAISS] = None

    def index(self):
        """Полная переиндексация ТОЛЬКО из S3."""
        docs = load_corpus_s3()
        self.vs = build_store(docs)
        return self

    def open(self):
        """Открывает уже существующий индекс из VSTORE_DIR."""
        self.vs = load_store()
        return self

    def retrieve(self, query: str, k: int = 4):
        if self.vs is None:
            self.open()
        return self.vs.as_retriever(search_kwargs={"k": k}).invoke(query)

    def context(self, query: str, k: int = 4, max_chars: int = 3000) -> str:
        docs = self.retrieve(query, k=k)
        text = "\n\n".join(d.page_content for d in docs)
        return text[:max_chars]

    def retrieve_with_sources(self, query: str, k: int = 4) -> List[dict]:
        """Для отладки: возвращает text + source + page."""
        docs = self.retrieve(query, k=k)
        out = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            out.append({
                "source": md.get("source", ""),
                "page": md.get("page"),
                "text": d.page_content,
            })
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RAG (S3-only) index builder")
    ap.add_argument("--reindex", action="store_true", help="Перестроить индекс из S3 и сохранить локально")
    args = ap.parse_args()

    if args.reindex:
        RAG().index().open()
        print("ok: индекс построен/обновлён из S3.")
    else:
        RAG().open()
        print("ok: индекс открыт.")
