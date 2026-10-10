from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pymupdf

from src.logging_config import logger


@dataclass(frozen=True)
class PageText:
    source: str
    page: int
    text: str
    tables: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    content_type: str = "text"
    table_index: int | None = None


def extract_pages(documents_dir: str | Path) -> list[PageText]:
    root = Path(documents_dir)
    pdfs = sorted(root.rglob("*.pdf"))
    logger.info("ingestion.start directory=%s pdf_count=%d", root, len(pdfs))
    pages: list[PageText] = []
    for pdf_path in pdfs:
        before = len(pages)
        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    tables = tuple(extract_tables(page))
                    pages.append(PageText(pdf_path.name, page_number, clean_text(text), tables))
                    if tables:
                        logger.info("ingestion.tables source=%s page=%d count=%d", pdf_path.name, page_number, len(tables))
        logger.info("ingestion.pdf source=%s text_pages=%d", pdf_path.name, len(pages) - before)
    logger.info("ingestion.complete text_pages=%d", len(pages))
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tables(page) -> list[str]:
    """Extract detected tables as Markdown while retaining their headers."""
    if not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
    except Exception as exc:  # table detection is best-effort for unusual PDFs
        logger.warning("ingestion.table_detection_failed page=%s error=%s", page.number + 1, exc)
        return []
    markdown_tables = []
    for table in finder.tables:
        rows = table.extract()
        if not rows or not rows[0]:
            continue
        width = len(rows[0])
        normalized = [[(cell or "").replace("|", "\\|").replace("\n", " ").strip() for cell in row] for row in rows]
        normalized = [row + [""] * (width - len(row)) for row in normalized]
        header = normalized[0]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
        markdown_tables.append("\n".join(lines))
    return markdown_tables


def chunk_pages(pages: list[PageText], chunk_size: int = 900, overlap: int = 150) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[TextChunk] = []
    for page in pages:
        words = page.text.split()
        start = 0
        local_index = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end])
            chunks.append(TextChunk(
                chunk_id=f"{page.source}:{page.page}:{local_index}",
                source=page.source,
                page_start=page.page,
                page_end=page.page,
                text=text,
            ))
            local_index += 1
            if end == len(words):
                break
            start = end - overlap
        for table_index, table_text in enumerate(page.tables, start=1):
            chunks.append(TextChunk(
                chunk_id=f"{page.source}:{page.page}:table:{table_index}",
                source=page.source,
                page_start=page.page,
                page_end=page.page,
                text=table_text,
                content_type="table",
                table_index=table_index,
            ))
    logger.info("chunking.complete pages=%d chunks=%d chunk_size=%d overlap=%d", len(pages), len(chunks), chunk_size, overlap)
    return chunks
