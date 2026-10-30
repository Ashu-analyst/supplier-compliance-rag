"""Small HTTP API so n8n (or any tool) can call the assistant with an HTTP Request node."""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from rag import HybridRetriever, answer, load_corpus

KB_PATH = os.getenv("KB_PATH", "knowledge_base")

app = FastAPI(title="Supplier Compliance RAG Assistant")
retriever = HybridRetriever(load_corpus(KB_PATH))


class Question(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(retriever.chunks)}


@app.post("/ask")
def ask(q: Question) -> dict:
    a = answer(q.question, retriever, k=q.k)
    return {"answer": a.text, "citations": a.citations, "refused": a.refused, "backend": a.backend}
