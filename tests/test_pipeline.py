from pathlib import Path

import pytest

from studyrag.chunking import chunk_documents
from studyrag.generate import answer, build_prompt
from studyrag.index import VectorIndex
from studyrag.loaders import Document, load_directory

NOTES = Path(__file__).parent.parent / "examples" / "notes"


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_chunking_overlap():
    doc = Document("a.txt", " ".join(str(i) for i in range(250)))
    chunks = chunk_documents([doc], chunk_size=100, overlap=20)
    assert len(chunks) == 3
    assert chunks[1].text.split()[0] == "80"


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_documents([Document("a", "x")], chunk_size=10, overlap=10)


def test_retrieval_finds_right_file(tmp_path):
    index = VectorIndex().build(chunk_documents(load_directory(NOTES), chunk_size=60, overlap=15))
    hits = index.search("What does a Trusted Platform Module do?", k=2)
    assert hits[0].chunk.source == "iot_security.md"

    path = tmp_path / "idx.joblib"
    index.save(path)
    assert VectorIndex.load(path).search("learning rate", k=1)[0].chunk.source == "machine_learning.md"


def test_extractive_answer_cites_sources():
    index = VectorIndex().build(chunk_documents(load_directory(NOTES)))
    hits = index.search("What is a container in Docker?", k=3)
    result = answer("What is a container in Docker?", hits)
    assert result.mode == "extractive"
    assert "container" in result.text.lower()
    assert "[1]" in result.text or "[2]" in result.text
    assert "reproducibility.md" in result.sources


def test_prompt_numbers_passages():
    index = VectorIndex().build(chunk_documents(load_directory(NOTES)))
    prompt = build_prompt("PUF?", index.search("PUF", k=2))
    assert "[1]" in prompt and "Question: PUF?" in prompt
