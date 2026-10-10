from types import SimpleNamespace

from src.generation.answer import format_context


def test_context_contains_source_and_page():
    result = SimpleNamespace(source="report.pdf", page_start=4, text="Evidence")
    assert format_context([result]) == "[Context 1] report.pdf, page 4\nEvidence"
