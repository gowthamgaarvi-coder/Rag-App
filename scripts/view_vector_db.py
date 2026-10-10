from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local Qdrant vector database")
    parser.add_argument("--limit", type=int, default=10, help="Number of chunks to display")
    parser.add_argument("--vectors", action="store_true", help="Display vector values")
    args = parser.parse_args()

    storage_path = Path(__file__).resolve().parents[1] / "data" / "qdrant_storage"
    collection_name = "pdf_chunks"
    client = QdrantClient(path=str(storage_path))

    print(f"Qdrant path: {storage_path}")
    print(f"Collections: {client.get_collections()}")
    print(f"Collection info: {client.get_collection(collection_name)}")

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=args.limit,
        with_payload=True,
        with_vectors=args.vectors,
    )

    print(f"Displaying {len(points)} chunks")
    for point in points:
        payload = point.payload or {}
        print("\n" + "=" * 80)
        print(f"ID: {point.id}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print(f"Source: {payload.get('source')}")
        print(f"Page: {payload.get('page_start')} - {payload.get('page_end')}")
        print(f"Content type: {payload.get('content_type', 'text')}")
        print(f"Table index: {payload.get('table_index')}")
        print("Text:")
        print(payload.get("text", ""))
        if args.vectors:
            print("Vector:")
            print(point.vector)


if __name__ == "__main__":
    main()
