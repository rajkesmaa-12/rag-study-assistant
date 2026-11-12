"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from studyrag.chunking import chunk_documents
from studyrag.generate import answer
from studyrag.index import VectorIndex
from studyrag.loaders import load_directory

st.set_page_config(page_title="StudyRAG", page_icon="📚")
st.title("📚 StudyRAG — ask your lecture notes")

files = st.file_uploader("Upload notes (.pdf, .md, .txt)", accept_multiple_files=True, type=["pdf", "md", "txt"])
if files and st.button("Build index"):
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            Path(tmp, f.name).write_bytes(f.getvalue())
        chunks = chunk_documents(load_directory(tmp))
    st.session_state.index = VectorIndex().build(chunks)
    st.success(f"Indexed {len(files)} files ({len(chunks)} chunks)")

if "index" in st.session_state:
    q = st.text_input("Your question")
    if q:
        hits = st.session_state.index.search(q, k=4)
        result = answer(q, hits)
        st.markdown(result.text)
        with st.expander(f"Sources · mode: {result.mode}"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h.chunk.source}** (score {h.score:.2f})\n\n{h.chunk.text}")
