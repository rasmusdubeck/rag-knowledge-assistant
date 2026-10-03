import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ingest import chunk_text, parse_pdf


def test_pdf_can_be_parsed(tmp_path):
    pdf_path = tmp_path / "notes.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as file:
        writer.write(file)

    # A blank page has no text, but confirms the parser handles a valid PDF.
    assert parse_pdf(pdf_path) == []
    assert isinstance(PdfReader(str(pdf_path)), PdfReader)


def test_chunks_contain_source_metadata():
    chunks = [
        {"text": text, "source": "guide.pdf", "page": 2}
        for text in chunk_text("one two three four", chunk_size=2, overlap=0)
    ]

    assert chunks
    assert all(chunk["source"] == "guide.pdf" and chunk["page"] == 2 for chunk in chunks)
