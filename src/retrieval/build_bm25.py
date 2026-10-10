from pathlib import Path
import os

from src.ingestion.pdf_loader import chunk_pages, extract_pages
from src.logging_config import logger
from src.retrieval.index_artifacts import save_artifacts


def main() -> None:
    documents_dir = Path(os.environ.get("DOCUMENTS_DIR", r"C:\Users\gowth\OneDrive\09-10-2026\Documents"))
    chunks = chunk_pages(extract_pages(documents_dir))
    save_artifacts(chunks)
    logger.info("artifacts.complete chunks=%d", len(chunks))


if __name__ == "__main__":
    main()
