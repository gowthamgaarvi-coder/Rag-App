from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts this file's directory on sys.path. The application packages
# live in the project root, so add that root explicitly before importing src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer import RAGAnswerer
from src.logging_config import logger


st.set_page_config(page_title="Hybrid PDF RAG", page_icon="📚", layout="wide")
st.title("Hybrid PDF RAG")
st.caption("Local embeddings + BM25 + BGE reranking + local Qwen3 generation")


@st.cache_resource
def get_answerer() -> RAGAnswerer:
    logger.info("streamlit.answerer.load")
    return RAGAnswerer()


question = st.text_area(
    "Ask a question about the PDFs",
    placeholder="For example: What was Acme Corp's FY2025 revenue?",
    height=100,
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching documents and generating an answer..."):
        result = get_answerer().answer(question.strip())
    st.subheader("Answer")
    st.write(result.text)
    st.subheader("Retrieved sources")
    for source in result.sources:
        with st.expander(
            f"{source.source} — page {source.page_start} — chunk {source.chunk_id} — "
            f"reranker {source.score:.4f} | cosine {source.cosine_similarity:.4f}"
        ):
            st.write(f"Chunk ID: `{source.chunk_id}`")
            st.write(f"Content source: `{source.source}`")
            st.write(f"Page: `{source.page_start}`")
            st.write(f"Reranker score: `{source.score:.4f}`")
            st.write(f"Cosine similarity: `{source.cosine_similarity:.4f}`")
            if "|" in source.text and source.text.count("|") >= 4:
                st.markdown(source.text)
            else:
                st.write(source.text)
    with st.expander("Pipeline activity"):
        st.caption("Detailed logs are written to logs/rag_app.log and the terminal.")
        try:
            log_file = PROJECT_ROOT / "logs" / "rag_app.log"
            lines = log_file.read_text(encoding="utf-8").splitlines()[-30:]
            st.code("\n".join(lines), language="text")
        except FileNotFoundError:
            st.info("No log file has been created yet.")
