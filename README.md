# RAG Knowledge Assistant

A small learning project that answers questions about local PDF documents with a
minimal Retrieval-Augmented Generation (RAG) pipeline.

## Pipeline

PDF documents are parsed with `pypdf`, split into chunks, embedded with
`sentence-transformers/all-MiniLM-L6-v2`, and stored in a local FAISS index.
The most relevant chunks are then sent as context to a configurable Gemini
Flash-Lite model, which answers the question and cites the supplied sources.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the Gemini key and, optionally, a different model:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
```

## Use

1. Put PDF files in `data/documents/`.
2. Build the local index:

   ```bash
   python src/ingest.py
   ```

3. Start the question-answering CLI:

   ```bash
   python src/main.py
   ```

Type a question, or `quit` to exit. The generated FAISS files are kept in
`data/index/` and are ignored by git.

## Technologies

Python, pypdf, sentence-transformers, FAISS, Google Gemini API, and pytest.
