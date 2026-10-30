# Supplier Compliance RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about supplier due-diligence procedures. Every answer cites its sources, and the assistant refuses when the knowledge base doesn't contain the answer. An evaluation harness measures retrieval quality, answer faithfulness and refusal behaviour on every commit.

> The knowledge base is a set of **synthetic** SOPs for a fictional company ("Northwind Components"). The domain is supplier risk, audits, grievances and ESG data collection: the kind of process knowledge that procurement and compliance teams ask about every day.

## Why this exists

Compliance teams answer the same questions again and again ("Who approves a supplier with a risk score of 55?", "How long does a supplier have for a corrective action plan?"). A wrong answer is worse than no answer, so the design goals are:

1. **Grounded**: answers come only from the documents, with citations.
2. **Honest**: if the question is out of scope, it says so and hands over to a human.
3. **Measured**: quality is a number in CI, not a feeling.

## Architecture

```
knowledge_base/*.md
      │  ingest.py   heading-aware chunking (title + section kept in every chunk)
      ▼
   chunks ──► BM25 (keywords)  ─┐
      │                         ├─► Reciprocal Rank Fusion ─► top-k passages
      └────► embeddings (dense) ┘                                 │
             lsa | sbert | ollama                                 ▼
                                             guardrail: coverage check ── refuse
                                                                  │
                                       Claude API | Ollama | extractive fallback
                                                                  ▼
                                                    answer + [n] citations
```

| Part | File | Choice and reason |
|---|---|---|
| Chunking | `rag/ingest.py` | Split on `##` headings, then 120-word windows with 20-word overlap. The title and section are prefixed so a chunk still makes sense on its own. |
| Keyword search | `rag/retrieve.py` | BM25. It's precise on exact terms like "CAP", "SAQ" and numbers. |
| Dense search | `rag/embeddings.py` | Pluggable. `lsa` (TF-IDF + SVD) runs offline; `sbert` and `ollama` (e.g. `nomic-embed-text`) give better semantics. |
| Fusion | `rag/retrieve.py` | Reciprocal Rank Fusion (k=60). It needs no score normalisation between the two retrievers. |
| Guardrail | `rag/answer.py` | Refuses when less than 30% of the question's IDF-weighted terms appear in the retrieved passages. |
| Generation | `rag/answer.py` | Claude (`ANTHROPIC_API_KEY`), a local model through Ollama (`OLLAMA_HOST`), or an extractive fallback. The prompt forces citations and an exact refusal sentence. |
| API | `app.py` | FastAPI `POST /ask`, so n8n or any tool can call it. |
| Automation | `n8n_workflow.json` | Webhook → RAG API → if refused, escalate to a human; else reply with sources. |

## Run it

```bash
pip install -r requirements.txt
python ask.py "Who approves a supplier with a risk score of 55?"
python eval/run_eval.py          # quality report
pytest -q                        # unit + API tests
uvicorn app:app --reload         # http://localhost:8000/docs
```

With a local LLM and the API in containers:

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.1
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How often are high risk suppliers audited?"}'
```

## Evaluation results

Golden set: 17 questions, of which 15 are answerable and 2 are out of scope.

| Metric | Offline mode (LSA + extractive) |
|---|---|
| hit@4 (right document retrieved) | **1.00** |
| MRR | **0.97** |
| Refusal accuracy | **1.00** |
| Citation rate | **1.00** |
| Fact recall (key fact in answer) | 0.73 |

**Reading the numbers:** retrieval and refusal are solid. Fact recall is the weak spot of the *extractive* fallback. It picks sentences and can't combine them, so for 4 of 15 questions the right passage was retrieved but the wrong sentence was chosen. That's exactly the job of the LLM step. Run `python eval/run_eval.py` with `ANTHROPIC_API_KEY` or `OLLAMA_HOST` set to measure the generative mode on the same set. CI fails the build if hit@4 or refusal accuracy drop below 0.9.

## What I would add for production

- A vector database (pgvector or Qdrant) and incremental re-indexing when documents change
- A re-ranker (cross-encoder) on the top 10 before generation
- A bigger golden set built from real user questions, plus LLM-as-judge scoring for faithfulness
- Access control per document (the retention SOP says grievance cases are Compliance-only)
- Logging of refused questions, which shows where the knowledge base has gaps

## Project structure

```
knowledge_base/   synthetic SOP documents (markdown)
rag/              ingest, embeddings, retrieval, answer + guardrail
eval/             golden_set.jsonl + run_eval.py
tests/            pytest: chunking, retrieval, citations, refusal, API
app.py            FastAPI service      ask.py   CLI
Dockerfile, docker-compose.yml (API + Ollama), .github/workflows/ci.yml
```
