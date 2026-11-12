# 📚 StudyRAG — Ask Your Own Lecture Notes

[![tests](https://github.com/rajkesmaa-12/rag-study-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/rajkesmaa-12/rag-study-assistant/actions/workflows/tests.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented generation (RAG) assistant that answers questions **from your own study material** (PDF, Markdown, text) and **cites the passages it used**.

Works fully offline with a TF-IDF retriever and an extractive answerer, and switches to **Claude** or **OpenAI** for generated answers when an API key is set.

## How it works

```
notes (.pdf/.md/.txt) ─► load ─► chunk (overlapping word windows)
                                   │
                                   ▼
                     embed: TF-IDF  or  sentence-transformers
                                   │
question ─► embed ─► cosine top-k ─┘
                │
                ▼
        LLM (Claude / OpenAI) with numbered context ─► answer + [1][2] citations
        └─ no API key? extractive answer from the best-matching sentences
```

## Quick start

```bash
pip install -r requirements.txt

# command line
python -m studyrag ingest examples/notes
python -m studyrag ask "What does a TPM do?"

# web UI
streamlit run app.py
```

Optional: put `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (see `.env.example`) for LLM-written answers. Use `--backend sbert` (after `pip install sentence-transformers`) for dense semantic retrieval.

### Example

```
$ python -m studyrag ask "What is ASCON?"
ASCON is a family of lightweight authenticated encryption algorithms selected by NIST for constrained devices. [1]

Sources (extractive): iot_security.md
```

## Evaluation

`eval/evaluate_retrieval.py` measures whether the retriever finds the correct source file for labelled questions (Hit@k, MRR).

| Retriever | Questions | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| TF-IDF (1–2 grams) | 8 | 1.00 | 1.00 | 1.00 |

> The bundled question set is small and only covers the example notes. It checks that the pipeline works; it is not a benchmark. Add your own questions to `eval/questions.json` for a real test.

```bash
python eval/evaluate_retrieval.py --k 3
```

## Project structure

```
studyrag/
├── loaders.py    # .pdf / .md / .txt → Document
├── chunking.py   # overlapping word-window chunks
├── index.py      # TF-IDF or sentence-transformer vector index (save/load)
├── generate.py   # Claude / OpenAI / extractive answer with citations
└── __main__.py   # CLI: ingest, ask
app.py            # Streamlit UI
eval/             # retrieval evaluation
tests/            # pytest unit tests
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Tech
Python · scikit-learn · sentence-transformers (optional) · Anthropic / OpenAI APIs · Streamlit · pytest

## Roadmap
- [ ] Hybrid retrieval (BM25 + dense) with re-ranking
- [ ] Answer-quality evaluation (faithfulness, answer relevance)
- [ ] Persist indexes per course
