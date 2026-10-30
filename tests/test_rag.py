from pathlib import Path

from fastapi.testclient import TestClient

from rag import HybridRetriever, answer, load_corpus
from rag.ingest import chunk_document

KB = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_chunks_keep_title_and_section():
    chunks = chunk_document(KB / "02_risk_scoring.md")
    assert chunks, "document should produce chunks"
    assert all(c.text.startswith("Risk Scoring Methodology") for c in chunks)
    assert {c.section for c in chunks} >= {"Components and weights", "Risk bands"}


def test_retrieval_finds_right_document():
    r = HybridRetriever(load_corpus(KB))
    top = r.search("How often are high risk suppliers audited?", k=3)
    assert top[0].chunk.doc_id == "04_audit_programme"


def test_answer_has_citation():
    r = HybridRetriever(load_corpus(KB))
    a = answer("What weight does country risk have?", r)
    assert not a.refused and a.citations and "30%" in a.text


def test_guardrail_refuses_out_of_scope():
    r = HybridRetriever(load_corpus(KB))
    a = answer("What is the parental leave policy?", r)
    assert a.refused


def test_api_endpoint(monkeypatch):
    monkeypatch.chdir(KB.parent)
    import app as app_module

    client = TestClient(app_module.app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.post("/ask", json={"question": "Who can see grievance cases?"}).json()
    assert "Compliance" in body["answer"]
