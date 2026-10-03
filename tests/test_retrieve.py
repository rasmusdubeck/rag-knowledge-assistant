import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieve import retrieve


def test_retrieval_returns_requested_number_of_results(tmp_path, monkeypatch):
    records = [
        {"text": "first", "source": "a.pdf", "page": 1},
        {"text": "second", "source": "b.pdf", "page": 2},
        {"text": "third", "source": "c.pdf", "page": 3},
    ]
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "documents.faiss"))
    (tmp_path / "chunks.json").write_text(json.dumps(records), encoding="utf-8")

    class FakeModel:
        def encode(self, *_args, **_kwargs):
            return np.array([[1, 0]], dtype="float32")

    monkeypatch.setattr("retrieve.SentenceTransformer", lambda _name: FakeModel())
    from retrieve import _load_resources

    _load_resources.cache_clear()
    results = retrieve("question", k=2, index_dir=tmp_path)

    assert len(results) == 2
    assert results[0]["source"] == "a.pdf"
