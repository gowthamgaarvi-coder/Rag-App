from src.ingestion.pdf_loader import PageText, chunk_pages, clean_text


def test_clean_text_repairs_hyphenated_line_breaks():
    assert clean_text("well-\nknown   term") == "wellknown term"


def test_chunking_preserves_source_and_page_metadata():
    pages = [PageText("example.pdf", 3, "one two three four five six")]
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)
    assert chunks[0].source == "example.pdf"
    assert chunks[0].page_start == 3
    assert chunks[0].text == "one two three four"
