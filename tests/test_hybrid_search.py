from src.retrieval.hybrid_search import _tokens


def test_tokens_are_lowercase_words():
    assert _tokens("Hybrid RAG, v2!") == ["hybrid", "rag", "v2"]
