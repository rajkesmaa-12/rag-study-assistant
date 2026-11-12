"""Split documents into overlapping word windows."""
from __future__ import annotations

from dataclasses import dataclass

from .loaders import Document


@dataclass
class Chunk:
    id: int
    source: str
    text: str


def chunk_documents(docs: list[Document], chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for doc in docs:
        words = doc.text.split()
        for start in range(0, max(len(words), 1), step):
            window = words[start : start + chunk_size]
            if not window:
                break
            chunks.append(Chunk(id=len(chunks), source=doc.source, text=" ".join(window)))
            if start + chunk_size >= len(words):
                break
    return chunks
