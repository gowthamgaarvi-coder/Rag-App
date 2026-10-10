from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pickle
import re

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.logging_config import logger
from src.retrieval.index_artifacts import load_artifacts, tokens


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    score: float
    cosine_similarity: float = 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class HybridSearcher:
    def __init__(self) -> None:
        self.model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        self.client = QdrantClient(path="data/qdrant_storage")
        self.collection = os.environ.get("QDRANT_COLLECTION", "pdf_chunks")
        chunks, self.bm25 = load_artifacts()
        logger.info("hybrid.init chunks=%d collection=%s", len(chunks), self.collection)
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_order = [chunk.chunk_id for chunk in chunks]

    def search(self, query: str, dense_k: int = 20, sparse_k: int = 20, limit: int = 8) -> list[SearchResult]:
        logger.info("hybrid.search.start query=%r dense_k=%d sparse_k=%d limit=%d", query, dense_k, sparse_k, limit)
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        dense = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=dense_k,
            with_payload=True,
        ).points
        dense_scores = {point.payload["chunk_id"]: float(point.score) for point in dense}
        sparse_scores = self.bm25.get_scores(tokens(query))
        sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:sparse_k]

        fused: dict[str, float] = {}
        rank_constant = 60
        for rank, point in enumerate(dense, start=1):
            fused[point.payload["chunk_id"]] = fused.get(point.payload["chunk_id"], 0) + 1 / (rank_constant + rank)
        for rank, index in enumerate(sparse_indices, start=1):
            chunk_id = self.chunk_order[index]
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (rank_constant + rank)

        results = []
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]:
            chunk = self.chunks[chunk_id]
            results.append(SearchResult(
                chunk_id, chunk.source, chunk.page_start, chunk.page_end, chunk.text,
                score, dense_scores.get(chunk_id, 0.0),
            ))
        logger.info("hybrid.search.complete dense_hits=%d sparse_hits=%d fused_hits=%d", len(dense), len(sparse_indices), len(results))
        return results
