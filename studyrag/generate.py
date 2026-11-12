"""Turn retrieved chunks into an answer.

Uses Claude (ANTHROPIC_API_KEY) or OpenAI (OPENAI_API_KEY) when a key is set.
Without a key it falls back to an extractive answer: the sentences from the
retrieved chunks that overlap most with the question.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .index import Hit

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the student's question using ONLY the numbered "
    "context passages. Cite passages like [1], [2]. If the context does not contain the "
    "answer, say you don't know."
)


@dataclass
class Answer:
    text: str
    sources: list[str]
    mode: str  # "claude", "openai" or "extractive"


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i}] ({h.chunk.source}) {h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    q = _tokens(question)
    scored = []
    for rank, h in enumerate(hits, 1):
        for sent in re.split(r"(?<=[.!?])\s+", h.chunk.text):
            overlap = len(q & _tokens(sent))
            if overlap:
                scored.append((overlap, -rank, f"{sent.strip()} [{rank}]"))
    if not scored:
        return "I couldn't find this in your notes."
    scored.sort(reverse=True)
    return " ".join(s for *_, s in scored[:max_sentences])


def answer(question: str, hits: list[Hit]) -> Answer:
    sources = list(dict.fromkeys(h.chunk.source for h in hits))
    if not hits:
        return Answer("I couldn't find this in your notes.", [], "extractive")
    prompt = build_prompt(question, hits)

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("STUDYRAG_CLAUDE_MODEL", "claude-sonnet-4-5"),
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return Answer(msg.content[0].text, sources, "claude")

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=os.getenv("STUDYRAG_OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        return Answer(resp.choices[0].message.content, sources, "openai")

    return Answer(extractive_answer(question, hits), sources, "extractive")
