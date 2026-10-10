# Hybrid PDF RAG Application

This project builds a local hybrid retrieval-augmented generation application for the PDFs in the configured documents directory.

## Pipeline

1. Extract PDF text and page metadata.
2. Clean and chunk the text.
3. Create local open-source embeddings.
4. Index dense vectors in Qdrant.
5. Build a BM25 sparse index.
6. Fuse dense and sparse retrieval results.
7. Rerank candidates locally.
8. Generate a cited answer with `gpt-5.6-luna`.
9. Evaluate retrieval and answer quality.

The implementation is being built step by step. See `docs/architecture.md` for the current design.

## Run the UI

1. Copy `.env.example` to `.env`.
2. Install and run a local model with Ollama:

```powershell
ollama pull qwen3:8b
```

Set `LLM_PROVIDER=ollama` in `.env`.
For the OpenAI fallback, set `LLM_PROVIDER=openai` and configure `OPENAI_API_KEY`.
3. From this directory, run:

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

The API can be started with:

```powershell
.\.venv\Scripts\uvicorn.exe app.api:app --reload
```

## Inspect the vector database

Stop Streamlit first if it is running, then display stored chunks:

```powershell
.\.venv\Scripts\python.exe scripts\view_vector_db.py --limit 10
```

To also display vector values:

```powershell
.\.venv\Scripts\python.exe scripts\view_vector_db.py --limit 1 --vectors
```
