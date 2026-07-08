import streamlit as st

from generate import generate_answer


st.set_page_config(page_title="AI Papers RAG Assistant", layout="wide")

st.title("AI Papers RAG Assistant")

question = st.text_input("Ask a question about the papers")

top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=5)

if st.button("Ask") and question:
    with st.spinner("Retrieving evidence and generating answer..."):
        answer, hits = generate_answer(question, top_k=top_k)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]

        with st.expander(
            f"[{i}] {metadata['paper_id']} | page {metadata['page']} | chunk {metadata['chunk_index']}"
        ):
            st.write(hit["text"])