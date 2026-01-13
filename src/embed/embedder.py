from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=normalize)
