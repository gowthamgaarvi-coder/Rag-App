from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.answer import RAGAnswerer


app = FastAPI(title="Hybrid PDF RAG API")
answerer: RAGAnswerer | None = None


def get_answerer() -> RAGAnswerer:
    global answerer
    if answerer is None:
        answerer = RAGAnswerer()
    return answerer


class Question(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict:
    result = get_answerer().answer(payload.question)
    return {
        "answer": result.text,
        "sources": [
            {
                "source": item.source,
                "page": item.page_start,
                "reranker_score": item.score,
                "cosine_similarity": item.cosine_similarity,
            }
            for item in result.sources
        ],
    }
