"""
FAISS vector DB — load, search, add, persist.

Expects an IndexFlatIP (inner product) index built from L2-normalized
embeddings so that IP == cosine similarity ∈ [-1, 1].
"""
import json
import os

import faiss
import numpy as np


class VectorDB:
    def __init__(self, index_path: str, labels_path: str | None = None):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index       = faiss.read_index(index_path)
        self._index_path = index_path

        lpath = labels_path or index_path + ".labels.json"
        if not os.path.exists(lpath):
            raise FileNotFoundError(f"Labels file not found: {lpath}")
        with open(lpath, encoding="utf-8") as f:
            self.labels: list[str] = json.load(f)
        self._labels_path = lpath

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal

    @property
    def num_words(self) -> int:
        return len(set(self.labels))

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return up to top_k (word, cosine_score) pairs, deduplicated by word."""
        q       = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        fetch_k = min(top_k * 10, self.index.ntotal)
        dists, idxs = self.index.search(q, fetch_k)

        best: dict[str, float] = {}
        for dist, idx in zip(dists[0], idxs[0]):
            if 0 <= idx < len(self.labels):
                word = self.labels[idx]
                if word not in best or dist > best[word]:
                    best[word] = float(dist)

        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, embedding: np.ndarray, label: str) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        self.index.add(vec)
        self.labels.append(label)

    def save(self) -> None:
        faiss.write_index(self.index, self._index_path)
        with open(self._labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
