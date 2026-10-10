from src.evaluation.metrics import evaluate_cases, load_cases
from src.retrieval.hybrid_search import HybridSearcher


def main() -> None:
    searcher = HybridSearcher()
    scores = evaluate_cases(
        load_cases(),
        retrieve=lambda question, k: searcher.search(question, limit=k),
        k=5,
    )
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
