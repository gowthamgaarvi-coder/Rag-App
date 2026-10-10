from src.ingestion.pdf_loader import PageText, chunk_pages, extract_tables


def test_table_chunk_has_table_metadata():
    chunks = chunk_pages([PageText("catalog.pdf", 2, "Description", ("| Model | Price |\n| --- | --- |\n| A | 10 |",))])
    table = next(chunk for chunk in chunks if chunk.content_type == "table")
    assert table.table_index == 1
    assert table.page_start == 2
    assert "| Model | Price |" in table.text
