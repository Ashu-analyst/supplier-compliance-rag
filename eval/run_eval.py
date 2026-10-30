"""Offline evaluation: retrieval quality + answer faithfulness + refusal behaviour.

Metrics
- hit@k        : expected document appears in the top-k retrieved chunks
- MRR          : mean reciprocal rank of the first chunk from the expected document
- fact recall  : answer text contains the key fact (only for answerable questions)
- citation rate: answer cites at least one passage (only for answerable questions)
- refusal acc. : unanswerable questions are refused AND answerable ones are not
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import HybridRetriever, answer, load_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(k: int = 4) -> dict:
    retriever = HybridRetriever(load_corpus(ROOT / "knowledge_base"))
    rows = [json.loads(l) for l in (ROOT / "eval" / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    hits, rr, facts, cites, refusal_ok, answerable = 0, 0.0, 0, 0, 0, 0
    failures = []
    for r in rows:
        res = answer(r["question"], retriever, k=k)
        if r["expected_doc"] is None:
            refusal_ok += res.refused
            if not res.refused:
                failures.append(("should refuse", r["question"], res.text[:120]))
            continue
        answerable += 1
        refusal_ok += not res.refused
        docs = [h.chunk.doc_id for h in retriever.search(r["question"], k=k)]
        if r["expected_doc"] in docs:
            hits += 1
            rr += 1 / (docs.index(r["expected_doc"]) + 1)
        ok_fact = all(f.lower() in res.text.lower() for f in r["must_contain"])
        facts += ok_fact
        cites += bool(res.citations)
        if not ok_fact:
            failures.append(("missing fact", r["question"], res.text[:120]))
    report = {
        "questions": len(rows),
        f"hit@{k}": round(hits / answerable, 3),
        "MRR": round(rr / answerable, 3),
        "fact_recall": round(facts / answerable, 3),
        "citation_rate": round(cites / answerable, 3),
        "refusal_accuracy": round(refusal_ok / len(rows), 3),
        "failures": failures,
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
