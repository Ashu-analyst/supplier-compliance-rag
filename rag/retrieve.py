"""Hybrid retrieval: BM25 (keyword) + dense embeddings, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder, get_embedder
from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "be",
    "by", "with", "what", "how", "when", "who", "which", "do", "does", "we", "our", "it",
    "must", "can", "that", "this", "from", "at", "as", "if",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    bm25_rank: int | None
    dense_rank: int | None
    dense_similarity: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, rrf_k: int = 60):
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.embedder = embedder or get_embedder(os.getenv("EMBEDDINGS", "lsa"))
        self.embedder.fit([c.text for c in chunks])
        self.matrix = self.embedder.embed([c.text for c in chunks])  # L2-normalised rows

    def search(self, query: str, k: int = 4, candidates: int = 10) -> list[Hit]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_order = list(np.argsort(-bm25_scores)[:candidates])

        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q  # cosine similarity because rows are normalised
        dense_order = list(np.argsort(-sims)[:candidates])

        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm25_order):
            if bm25_scores[idx] > 0:
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense_order):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=fused[i],
                bm25_rank=bm25_order.index(i) + 1 if i in bm25_order else None,
                dense_rank=dense_order.index(i) + 1 if i in dense_order else None,
                dense_similarity=float(sims[i]),
            )
            for i in best
        ]
