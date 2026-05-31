import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


RAG_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/rag_index"))
RAG_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = RAG_DIR / "chunks.jsonl"
META_PATH = RAG_DIR / "metadata.json"
EMBEDDINGS_PATH = RAG_DIR / "embeddings.npy"

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float
    rank: int


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            end = start + chunk_size
            final_chunks.append(chunk[start:end].strip())
            start = max(end - overlap, end)

    return [c for c in final_chunks if c]


def load_text_from_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".json", ".jsonl", ".py"}:
        return content.decode("utf-8", errors="ignore")

    return content.decode("utf-8", errors="ignore")


def _load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def _save_chunks(chunks: list[dict[str, Any]]) -> None:
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _save_metadata(metadata: dict[str, Any]) -> None:
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_metadata() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}

    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _get_sentence_model():
    if SentenceTransformer is None:
        return None

    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def _embed_texts(texts: list[str]) -> tuple[np.ndarray | None, str]:
    model = _get_sentence_model()

    if model is None:
        return None, "tfidf"

    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32), DEFAULT_EMBEDDING_MODEL


def rebuild_embeddings() -> dict[str, Any]:
    chunks = _load_chunks()

    if not chunks:
        if EMBEDDINGS_PATH.exists():
            EMBEDDINGS_PATH.unlink()

        metadata = {
            "status": "empty",
            "chunks": 0,
            "embedding_backend": None,
            "updated_at": time.time(),
        }
        _save_metadata(metadata)
        return metadata

    texts = [c["text"] for c in chunks]
    embeddings, backend = _embed_texts(texts)

    if embeddings is not None:
        np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "status": "ready",
        "chunks": len(chunks),
        "sources": sorted({c["source"] for c in chunks}),
        "embedding_backend": backend,
        "updated_at": time.time(),
    }
    _save_metadata(metadata)

    return metadata


def ingest_document(filename: str, content: bytes) -> dict[str, Any]:
    text = load_text_from_upload(filename, content)
    chunks = chunk_text(text)

    existing = _load_chunks()
    source_id = f"{Path(filename).stem}-{uuid.uuid4().hex[:8]}"

    new_chunks = []
    for i, chunk in enumerate(chunks):
        new_chunks.append(
            {
                "chunk_id": f"{source_id}-{i}",
                "source": filename,
                "source_id": source_id,
                "chunk_index": i,
                "text": chunk,
                "chars": len(chunk),
            }
        )

    all_chunks = existing + new_chunks
    _save_chunks(all_chunks)
    metadata = rebuild_embeddings()

    return {
        "filename": filename,
        "source_id": source_id,
        "chunks_added": len(new_chunks),
        "total_chunks": metadata.get("chunks", len(all_chunks)),
        "embedding_backend": metadata.get("embedding_backend"),
    }


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    chunks = _load_chunks()
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    metadata = _load_metadata()
    backend = metadata.get("embedding_backend")

    if backend and backend != "tfidf" and EMBEDDINGS_PATH.exists() and SentenceTransformer is not None:
        model = _get_sentence_model()
        query_embedding = model.encode([query], normalize_embeddings=True)
        doc_embeddings = np.load(EMBEDDINGS_PATH)

        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1])[0]

    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked, start=1):
        chunk = chunks[int(idx)]
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"],
                text=chunk["text"],
                score=round(float(scores[int(idx)]) * 100, 2),
                rank=rank,
            )
        )

    return results


def build_context(retrieved: list[RetrievedChunk]) -> str:
    parts = []

    for item in retrieved:
        parts.append(
            f"[Source: {item.source} | Chunk: {item.chunk_id} | Score: {item.score}]\n{item.text}"
        )

    return "\n\n".join(parts)


def index_status() -> dict[str, Any]:
    metadata = _load_metadata()
    chunks = _load_chunks()

    return {
        "status": metadata.get("status", "empty" if not chunks else "ready"),
        "chunks": len(chunks),
        "sources": sorted({c["source"] for c in chunks}),
        "embedding_backend": metadata.get("embedding_backend", "not_built"),
        "index_dir": str(RAG_DIR),
    }


def reset_index() -> dict[str, Any]:
    for path in [CHUNKS_PATH, META_PATH, EMBEDDINGS_PATH]:
        if path.exists():
            path.unlink()

    return index_status()