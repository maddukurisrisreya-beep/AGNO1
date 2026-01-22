import streamlit as st
import os

from embeddings.embedder import Embedder
from vectorstore.store import VectorStore
from utils.chunking import chunk_by_section
from tools.tool import init_tools
from agents.agent import create_resume_agent
from resume_parser import extract_resume_text


st.set_page_config(page_title="Agentic Resume RAG", layout="centered")
st.title("📄 Agentic Resume RAG Chatbot")


embedder = Embedder()
agent = None


uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    os.makedirs("data/resumes", exist_ok=True)
    resume_path = f"data/resumes/{uploaded_file.name}"

    with open(resume_path, "wb") as f:
        f.write(uploaded_file.read())

    resume_text = extract_resume_text(resume_path)
    chunks = chunk_by_section(resume_text)

    embeddings = embedder.embed(chunks)

    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    init_tools(embedder, vector_store)
    agent = create_resume_agent()

    st.success("Resume processed successfully!")


if agent:
    question = st.text_input("Ask a question about the resume")

    if question:
        response = agent.run(question)
        st.subheader("Answer")
        st.write(response.get_content_as_string())
