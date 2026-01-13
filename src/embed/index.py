from __future__ import annotations

from typing import List

import faiss
import numpy as np


def build_faiss(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return index


def search_faiss(
    index: faiss.IndexFlatIP,
    vectors: np.ndarray,
    _embedder,
    queries: List[str],
    top_k: int,
) -> List[List[int]]:
    query_vecs = _embedder.encode(queries, normalize=True).astype("float32")
    scores, indices = index.search(query_vecs, top_k)
    return indices.tolist()
