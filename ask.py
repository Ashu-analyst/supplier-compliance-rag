"""Command-line entry point: python ask.py "What happens when a supplier scores 75?" """
import sys

from rag import HybridRetriever, answer, load_corpus

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who approves a supplier with a risk score of 55?"
    retriever = HybridRetriever(load_corpus("knowledge_base"))
    result = answer(question, retriever)
    print(f"Q: {question}\nA: {result.text}\n")
    print(f"Backend: {result.backend} | Refused: {result.refused}")
    for c in result.citations:
        print(f"  source: {c}")
