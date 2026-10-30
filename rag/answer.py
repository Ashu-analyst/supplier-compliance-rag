"""Answer generation with citations and a 'no answer' guardrail.

LLM backend is chosen by environment:
- ANTHROPIC_API_KEY set  -> Claude via the Messages API
- OLLAMA_HOST set        -> local model via Ollama (e.g. llama3.1)
- neither                -> extractive mode: returns the most relevant sentences with citations

The guardrail refuses to answer when retrieval confidence is too low, instead of letting
the model guess. This is the single most important control in a compliance assistant.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass

from .retrieve import Hit, HybridRetriever, tokenize

NO_ANSWER = "I could not find this in the knowledge base. Please ask the Compliance team."

SYSTEM_PROMPT = (
    "You answer questions about Northwind's supplier compliance procedures. "
    "Use ONLY the numbered context passages. Cite every claim with its passage number like [1]. "
    "If the context does not contain the answer, reply exactly: " + NO_ANSWER
)


@dataclass
class Answer:
    text: str
    citations: list[str]
    refused: bool
    backend: str


def _context_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.chunk.chunk_id}) {h.chunk.text}" for i, h in enumerate(hits))


def _call_claude(question: str, context: str) -> str:
    body = {
        "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        "max_tokens": 400,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["content"][0]["text"]


def _call_ollama(question: str, context: str) -> str:
    body = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    }
    req = urllib.request.Request(
        f"{os.environ['OLLAMA_HOST']}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]


_QUANT = re.compile(r"^(how (many|long|often|fast|much|soon)|within|what is the deadline|what weight|by how)", re.I)


# Words that describe the *kind* of question, not its topic. They are ignored when we
# measure whether the knowledge base covers a question.
QUESTION_WORDS = {
    "how", "many", "much", "long", "fast", "often", "soon", "deadline", "give", "take",
    "get", "need", "have", "has", "within", "there", "their", "they", "new", "before",
    "after", "place", "send", "keep", "next", "who", "whom", "where", "why",
}


def _stem(tokens: list[str]) -> set[str]:
    """Very light prefix stemmer so 'approves'/'approval' and 'scores'/'score' match."""
    return {t[:5] for t in tokens if t not in QUESTION_WORDS}


def _numeric_fit(question: str, sentence: str) -> float:
    """Bonus when a number in the question falls inside a range stated in the sentence."""
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", question)]
    if not nums:
        return 0.0
    n = nums[0]
    for lo, hi in re.findall(r"(\d+) to (\d+)", sentence):
        if int(lo) <= n <= int(hi):
            return 2.0
    m = re.search(r"below (\d+)", sentence)
    if m and n < int(m.group(1)):
        return 2.0
    m = re.search(r"(\d+) or above", sentence)
    if m and n >= int(m.group(1)):
        return 2.0
    return 0.0


def _sentences(text: str) -> list[str]:
    body = text.split(": ", 1)[-1].replace("**", "")
    parts = re.split(r"(?<=[.!?])\s+|\s(?=- )|\s(?=\d+\.\s)", body)
    return [p.strip(" -") for p in parts if len(p.split()) >= 3]


def _extractive(question: str, hits: list[Hit], idf: dict[str, float]) -> str:
    """Rank sentences by IDF-weighted term overlap plus question-type cues; return the best
    sentence and the one that follows it in the same passage (procedures often span two)."""
    q = _stem(tokenize(question))
    wants_number = bool(_QUANT.search(question.strip()))
    best = None
    for rank, h in enumerate(hits):
        sents = _sentences(h.chunk.text)
        for j, sent in enumerate(sents):
            overlap = q & _stem(tokenize(sent))
            score = sum(idf.get(t, 1.0) for t in overlap)
            if wants_number and re.search(r"\d+\s*(working days|days|months|years|weeks|%|points)", sent):
                score += 2.0
            elif wants_number and re.search(r"\d", sent):
                score += 1.0
            # the section heading is context for every sentence below it
            overlap_heading = q & _stem(tokenize(h.chunk.section)) - overlap
            score += 0.5 * sum(idf.get(t, 1.0) for t in overlap_heading)
            if question.lower().startswith("who") and re.search(r"Manager|Officer|Committee|team|Legal", sent):
                score += 1.0
            score += _numeric_fit(question, sent)
            score -= 0.1 * rank  # small preference for higher-ranked passages
            if score > 0 and (best is None or score > best[0]):
                best = (score, rank, j, sents)
    if best is None:
        return NO_ANSWER
    _, rank, j, sents = best
    picked = sents[j : j + 2]
    return " ".join(picked) + f" [{rank + 1}]"


def _coverage(question: str, hits: list[Hit], idf: dict[str, float], unseen_idf: float) -> float:
    """Share of the question's (IDF-weighted) meaning that appears in the retrieved passages."""
    q = _stem(tokenize(question))
    if not q:
        return 0.0
    found = set().union(*(_stem(tokenize(h.chunk.text)) for h in hits)) if hits else set()
    total = sum(idf.get(t, unseen_idf) for t in q)
    return sum(idf.get(t, unseen_idf) for t in q & found) / total


def _idf(retriever: HybridRetriever) -> tuple[dict[str, float], float]:
    if not hasattr(retriever, "_idf_cache"):
        n = len(retriever.chunks)
        df: dict[str, int] = {}
        for c in retriever.chunks:
            for t in _stem(tokenize(c.text)):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        retriever._idf_cache = (idf, math.log(n + 1) + 1)  # unseen terms get the max IDF
    return retriever._idf_cache


def answer(question: str, retriever: HybridRetriever, k: int = 4, min_coverage: float = 0.3) -> Answer:
    hits = retriever.search(question, k=k)
    idf, unseen = _idf(retriever)

    # Guardrail: if most of the question's meaning is not in the knowledge base, refuse
    # instead of letting the model guess (e.g. HR or market-price questions).
    if not hits or _coverage(question, hits, idf, unseen) < min_coverage:
        return Answer(NO_ANSWER, [], True, "guardrail")

    context = _context_block(hits)
    if os.getenv("ANTHROPIC_API_KEY"):
        text, backend = _call_claude(question, context), "claude"
    elif os.getenv("OLLAMA_HOST"):
        text, backend = _call_ollama(question, context), "ollama"
    else:
        text, backend = _extractive(question, hits, idf), "extractive"

    refused = text.strip().startswith(NO_ANSWER)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
    return Answer(text, [hits[n - 1].chunk.chunk_id for n in cited], refused, backend)
