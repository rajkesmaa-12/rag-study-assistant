"""Command line interface.

    python -m studyrag ingest examples/notes
    python -m studyrag ask "What does a TPM do?"
"""
from __future__ import annotations

import argparse

from .chunking import chunk_documents
from .generate import answer
from .index import VectorIndex
from .loaders import load_directory

DEFAULT_INDEX = "studyrag_index.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(prog="studyrag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index a folder of notes")
    ing.add_argument("folder")
    ing.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    ing.add_argument("--index", default=DEFAULT_INDEX)

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=4)
    ask.add_argument("--index", default=DEFAULT_INDEX)

    args = parser.parse_args()
    if args.cmd == "ingest":
        docs = load_directory(args.folder)
        chunks = chunk_documents(docs)
        VectorIndex(args.backend).build(chunks).save(args.index)
        print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks -> {args.index}")
    else:
        index = VectorIndex.load(args.index)
        hits = index.search(args.question, k=args.k)
        result = answer(args.question, hits)
        print(result.text)
        print(f"\nSources ({result.mode}): " + ", ".join(result.sources))


if __name__ == "__main__":
    main()
