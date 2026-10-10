# Architecture

## Scope

The application answers questions over local PDF documents and exposes source metadata with every answer.

## Components

- **Ingestion:** PyMuPDF extracts text page by page.
- **Chunking:** recursive, structure-aware chunks retain document and page metadata.
- **Dense retrieval:** local `BAAI/bge-small-en-v1.5` embeddings stored in Qdrant.
- **Sparse retrieval:** BM25 over the same chunks.
- **Hybrid retrieval:** weighted reciprocal rank fusion combines dense and BM25 rankings.
- **Reranking:** local `BAAI/bge-reranker-base` scores the fused candidates.
- **Generation:** OpenAI `gpt-5.6-luna` receives only the selected context and must cite sources.
- **Evaluation:** retrieval metrics and answer-grounding metrics are recorded for a fixed test set.

## Data flow

```text
PDF → page text → chunks → embeddings/Qdrant + BM25
Question → dense search + BM25 → fusion → reranker → LLM → cited answer
```

## Planned directories

- `src/ingestion`: PDF extraction and chunking
- `src/retrieval`: embeddings, Qdrant, BM25, fusion, reranking
- `src/generation`: OpenAI client and answer prompt
- `src/evaluation`: dataset and metrics
- `app`: API and UI entry points
- `tests`: unit and integration tests
