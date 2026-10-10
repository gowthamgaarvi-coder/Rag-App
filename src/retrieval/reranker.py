from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.logging_config import logger


class RerankedSearcher:
    def __init__(self) -> None:
        self.hybrid = HybridSearcher()
        self.model = CrossEncoder(os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    def search(self, query: str, candidate_k: int = 20, limit: int = 5) -> list[SearchResult]:
        logger.info("reranker.start candidate_k=%d limit=%d", candidate_k, limit)
        candidates = self.hybrid.search(query, limit=candidate_k)
        if not candidates:
            return []
        scores = self.model.predict([(query, candidate.text) for candidate in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        results = [
            SearchResult(
                chunk_id=result.chunk_id,
                source=result.source,
                page_start=result.page_start,
                page_end=result.page_end,
                text=result.text,
                score=float(score),
                cosine_similarity=result.cosine_similarity,
            )
            for result, score in ranked[:limit]
        ]
        logger.info("reranker.complete candidates=%d results=%d", len(candidates), len(results))
        return results


if __name__ == "__main__":
    searcher = RerankedSearcher()
    results = searcher.search("What are the main findings in the documents?", limit=3)
    for result in results:
        print(f"{result.score:.4f} | {result.source} p.{result.page_start}")
