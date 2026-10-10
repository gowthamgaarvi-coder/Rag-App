from __future__ import annotations

from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi

from src.ingestion.pdf_loader import TextChunk


ARTIFACT_DIR = Path("data/indexes")
CHUNKS_PATH = ARTIFACT_DIR / "chunks.pkl"
BM25_PATH = ARTIFACT_DIR / "bm25.pkl"


def tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def save_artifacts(chunks: list[TextChunk]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("wb") as file:
        pickle.dump(chunks, file)
    bm25 = BM25Okapi([tokens(chunk.text) for chunk in chunks])
    with BM25_PATH.open("wb") as file:
        pickle.dump(bm25, file)


def load_artifacts() -> tuple[list[TextChunk], BM25Okapi]:
    if not CHUNKS_PATH.exists() or not BM25_PATH.exists():
        raise FileNotFoundError("Index artifacts are missing; run build_index first")
    with CHUNKS_PATH.open("rb") as file:
        chunks = pickle.load(file)
    with BM25_PATH.open("rb") as file:
        bm25 = pickle.load(file)
    return chunks, bm25
