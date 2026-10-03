"""Retrieve the most relevant chunks from the local FAISS index."""

import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from ingest import CHUNKS_FILE, DEFAULT_EMBEDDING_MODEL, INDEX_DIR, INDEX_FILE


@lru_cache(maxsize=1)
def _load_resources(index_dir: str, embedding_model_name: str):
    index = faiss.read_index(str(Path(index_dir) / INDEX_FILE.name))
    chunks = json.loads(
        (Path(index_dir) / CHUNKS_FILE.name).read_text(encoding="utf-8")
    )
    model = SentenceTransformer(embedding_model_name)
    return index, chunks, model


def retrieve(
    query: str,
    k: int = 5,
    index_dir: Path = INDEX_DIR,
    embedding_model_name: str | None = None,
) -> list[dict]:
    """Return up to k chunks, ordered by vector similarity."""
    if k < 1:
        raise ValueError("k must be at least 1")
    model_name = embedding_model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    index, chunks, model = _load_resources(str(index_dir), model_name)
    count = min(k, index.ntotal)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, positions = index.search(query_embedding, count)
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0:
            result = dict(chunks[position])
            result["score"] = float(score)
            results.append(result)
    return results
