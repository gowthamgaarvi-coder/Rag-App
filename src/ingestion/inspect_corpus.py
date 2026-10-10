from __future__ import annotations

import os

from .pdf_loader import chunk_pages, extract_pages


def main() -> None:
    documents_dir = os.environ.get(
        "DOCUMENTS_DIR",
        r"C:\Users\gowth\OneDrive\09-10-2026\Documents",
    )
    pages = extract_pages(documents_dir)
    chunks = chunk_pages(pages)
    print(f"PDF pages with text: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print("Sources:")
    for source in sorted({page.source for page in pages}):
        page_count = sum(page.source == source for page in pages)
        print(f"  - {source}: {page_count} pages")
    if chunks:
        print(f"First chunk: {chunks[0].chunk_id} ({len(chunks[0].text.split())} words)")


if __name__ == "__main__":
    main()
