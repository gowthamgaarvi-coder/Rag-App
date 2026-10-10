from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.ingestion.pdf_loader import chunk_pages, extract_pages
from src.retrieval.index_artifacts import save_artifacts


def main() -> None:
    documents_dir = Path(os.environ.get(
        "DOCUMENTS_DIR",
        r"C:\Users\gowth\OneDrive\09-10-2026\Documents",
    ))
    storage_dir = Path("data/qdrant_storage")
    collection_name = os.environ.get("QDRANT_COLLECTION", "pdf_chunks")
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    pages = extract_pages(documents_dir)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError(f"No text chunks found in {documents_dir}")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = QdrantClient(path=str(storage_dir))
    dimension = len(vectors[0])
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                    "content_type": chunk.content_type,
                    "table_index": chunk.table_index,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ],
    )
    save_artifacts(chunks)
    print(f"Indexed {len(chunks)} chunks")
    print(f"Embedding dimension: {dimension}")
    print("Saved chunks and BM25 artifacts to data/indexes")
    print(f"Qdrant path: {storage_dir.resolve()}")


if __name__ == "__main__":
    main()
