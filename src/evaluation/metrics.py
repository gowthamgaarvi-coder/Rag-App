from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _relevance(source: str, relevant_sources: set[str]) -> int:
    return int(source in relevant_sources)


def _unique_sources(retrieved: list[str]) -> list[str]:
    return list(dict.fromkeys(retrieved))


def precision_at_k(retrieved: list[str], relevant_sources: set[str], k: int) -> float:
    ranked = _unique_sources(retrieved)[:k]
    return sum(_relevance(source, relevant_sources) for source in ranked) / max(len(ranked), 1)


def recall_at_k(retrieved: list[str], relevant_sources: set[str], k: int) -> float:
    ranked = _unique_sources(retrieved)[:k]
    return sum(_relevance(source, relevant_sources) for source in ranked) / max(len(relevant_sources), 1)


def reciprocal_rank(retrieved: list[str], relevant_sources: set[str]) -> float:
    for rank, source in enumerate(_unique_sources(retrieved), start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant_sources: set[str], k: int) -> float:
    ranked = _unique_sources(retrieved)[:k]
    dcg = sum(_relevance(source, relevant_sources) / __import__("math").log2(rank + 1) for rank, source in enumerate(ranked, start=1))
    ideal_hits = min(len(relevant_sources), k)
    idcg = sum(1 / __import__("math").log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_cases(cases: Iterable[dict], retrieve, k: int = 5) -> dict[str, float]:
    rows = []
    for case in cases:
        results = retrieve(case["question"], k)
        retrieved = [result.source for result in results]
        relevant = set(case["relevant_sources"])
        rows.append({
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
        })
    if not rows:
        return {}
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]}


def load_cases(path: str | Path = Path(__file__).with_name("eval_questions.json")) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
