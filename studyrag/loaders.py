"""Load .txt, .md and .pdf files into plain-text documents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    source: str
    text: str


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so pdf support stays optional

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")
    return Document(source=path.name, text=text)


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No .txt/.md/.pdf files found in {directory}")
    return [load_file(p) for p in files]
