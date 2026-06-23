import json
import os
from typing import List, Tuple

import faiss
import numpy as np


class VectorDBService:
    def __init__(self, index_path: str, labels_path: str | None = None):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run 'python build_index.py' first to build the index."
            )
        self.index = faiss.read_index(index_path)
        self._index_path = index_path

        if labels_path is None:
            labels_path = index_path + ".labels.json"
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self._labels_path = labels_path

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal

    @property
    def num_words(self) -> int:
        return len(set(self.labels))

    def search(
        self, query_emb: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]

        # Fetch extra so that after dedup we still have top_k unique words
        fetch_k = min(top_k * 10, self.index.ntotal)
        distances, indices = self.index.search(q, fetch_k)

        # Keep best score per word (handles multiple vectors per word)
        best: dict[str, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.labels):
                word  = self.labels[idx]
                score = float(dist)
                if word not in best or score > best[word]:
                    best[word] = score

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_raw(self, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """중복 허용 raw top-k 검색."""
        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        fetch_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, fetch_k)
        return [
            (self.labels[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0])
            if 0 <= idx < len(self.labels)
        ]

    def add(self, embedding: np.ndarray, label: str) -> None:
        """Add one L2-normalized vector to the index."""
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        self.index.add(vec)
        self.labels.append(label)

    def save(self, index_path: str | None = None, labels_path: str | None = None) -> None:
        """Persist index and labels to disk."""
        ipath = index_path or self._index_path
        lpath = labels_path or self._labels_path
        faiss.write_index(self.index, ipath)
        with open(lpath, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
