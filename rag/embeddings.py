"""Pluggable embedding backends.

- "lsa"     : TF-IDF + truncated SVD (latent semantic analysis). Runs fully offline, no model download.
- "sbert"   : sentence-transformers (e.g. all-MiniLM-L6-v2). Better semantics, needs the model download.
- "ollama"  : local embedding model served by Ollama (e.g. nomic-embed-text), via OLLAMA_HOST.

All backends return L2-normalised vectors so a dot product equals cosine similarity.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    def __init__(self, dims: int = 64):
        self.dims = dims
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> None:
        x = self.tfidf.fit_transform(texts)
        n_comp = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(x)

    def embed(self, texts: list[str]) -> np.ndarray:
        assert self.svd is not None, "call fit() first"
        return _normalise(self.svd.transform(self.tfidf.transform(texts)))


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:  # pre-trained, nothing to fit
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(self.model.encode(texts)))


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            req = urllib.request.Request(
                f"{self.host}/api/embeddings",
                data=json.dumps({"model": self.model, "prompt": t}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                vectors.append(json.loads(r.read())["embedding"])
        return _normalise(np.asarray(vectors, dtype=float))


def get_embedder(name: str) -> Embedder:
    name = name.lower()
    if name == "lsa":
        return LSAEmbedder()
    if name == "sbert":
        return SentenceTransformerEmbedder()
    if name == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"Unknown embeddings backend: {name}")
