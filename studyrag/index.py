"""Vector index over chunks.

Two embedding back-ends:
* tfidf  – scikit-learn TF-IDF (default, no downloads, works offline)
* sbert  – sentence-transformers dense embeddings (better semantic recall)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, backend: str = "tfidf", model_name: str = "all-MiniLM-L6-v2"):
        if backend not in {"tfidf", "sbert"}:
            raise ValueError("backend must be 'tfidf' or 'sbert'")
        self.backend = backend
        self.model_name = model_name
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._encoder = None

    # ---- embedding -------------------------------------------------------
    def _sbert(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _embed_queries(self, texts: list[str]):
        if self.backend == "tfidf":
            return self._vectorizer.transform(texts)
        return self._sbert().encode(texts, normalize_embeddings=True)

    # ---- build / search --------------------------------------------------
    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if self.backend == "tfidf":
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform(texts)
        else:
            self._matrix = self._sbert().encode(texts, normalize_embeddings=True)
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self._embed_queries([query])
        scores = (self._matrix @ q.T)
        scores = np.asarray(scores.toarray() if hasattr(scores, "toarray") else scores).ravel()
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        encoder, self._encoder = self._encoder, None  # don't pickle the model
        joblib.dump(self, path)
        self._encoder = encoder

    @staticmethod
    def load(path: str | Path) -> "VectorIndex":
        return joblib.load(path)
