from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics():
    retrieved = ["wrong.pdf", "right.pdf", "other.pdf"]
    relevant = {"right.pdf"}
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) < 1
