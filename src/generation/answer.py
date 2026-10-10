from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.reranker import RerankedSearcher, SearchResult
from src.logging_config import logger

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer only from the supplied context. If the context does not contain enough
evidence, say that the documents do not provide enough information.
Do not invent facts, sources, page numbers, or citations.
For every factual claim, cite the supporting source using [filename, p. N].
Keep the answer concise but explain the reasoning when useful."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchResult]


def format_context(results: list[SearchResult]) -> str:
    sections = []
    for index, result in enumerate(results, start=1):
        sections.append(
            f"[Context {index}] {result.source}, page {result.page_start}\n{result.text}"
        )
    return "\n\n".join(sections)


class RAGAnswerer:
    def __init__(self, searcher: RerankedSearcher | None = None) -> None:
        self.searcher = searcher or RerankedSearcher()
        self.provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        if self.provider == "ollama":
            self.model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
            self.client = OpenAI(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
            )
        elif self.provider == "openai":
            self.model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'")

    def answer(self, question: str, limit: int = 5) -> Answer:
        logger.info("generation.start provider=%s model=%s question=%r", self.provider, self.model, question)
        results = self.searcher.search(question, limit=limit)
        context = format_context(results)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=f"Question:\n{question}\n\nRetrieved context:\n{context}",
        )
        logger.info("generation.complete sources=%d answer_chars=%d", len(results), len(response.output_text))
        return Answer(text=response.output_text, sources=results)
