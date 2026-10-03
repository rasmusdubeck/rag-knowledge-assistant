"""Build a FAISS index from PDF files in data/documents."""

import json
import os
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "documents.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_pdf(path: Path) -> list[dict]:
    """Extract non-empty page text and its 1-based page number from a PDF."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": path.name, "page": page_number})
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_index(
    documents_dir: Path = DOCUMENTS_DIR,
    index_dir: Path = INDEX_DIR,
    embedding_model_name: str | None = None,
) -> int:
    """Parse PDFs, create embeddings, and write the FAISS index and metadata."""
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {documents_dir}")

    records = []
    for pdf_path in pdf_paths:
        for page in parse_pdf(pdf_path):
            for text in chunk_text(page["text"]):
                records.append(
                    {"text": text, "source": page["source"], "page": page["page"]}
                )
    if not records:
        raise ValueError("No text could be extracted from the PDF files")

    model_name = embedding_model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [record["text"] for record in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE.name))
    (index_dir / CHUNKS_FILE.name).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(records)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks in {INDEX_DIR}")
