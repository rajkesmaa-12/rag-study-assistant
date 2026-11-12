"""Hit@k and MRR of the retriever on a small labelled question set.

    python eval/evaluate_retrieval.py --notes examples/notes --k 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studyrag.chunking import chunk_documents
from studyrag.index import VectorIndex
from studyrag.loaders import load_directory


def evaluate(notes: str, questions: str, k: int, backend: str) -> dict:
    index = VectorIndex(backend).build(chunk_documents(load_directory(notes), chunk_size=60, overlap=15))
    items = json.loads(Path(questions).read_text())
    hits, rr = 0, 0.0
    for item in items:
        sources = [h.chunk.source for h in index.search(item["question"], k=k)]
        if item["source"] in sources:
            hits += 1
            rr += 1 / (sources.index(item["source"]) + 1)
    return {"n": len(items), f"hit@{k}": hits / len(items), "mrr": rr / len(items)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notes", default="examples/notes")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--backend", default="tfidf", choices=["tfidf", "sbert"])
    a = p.parse_args()
    print(evaluate(a.notes, a.questions, a.k, a.backend))
