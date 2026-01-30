import os
import streamlit as st
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from vectorstore.store import VectorStore
from utils.chunking import chunk_by_section
from tools.tool import init_tools
from agents.agent import create_resume_agent
from resume_parser import extract_resume_text


# --------------------------------------------------
# LOAD ENV VARIABLES (LOCAL / DEPLOYMENT)
# --------------------------------------------------
load_dotenv(dotenv_path=".env", override=True)

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not set")
    st.stop()


# --------------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------------
st.set_page_config(page_title="Agentic Resume RAG", layout="centered")
st.title("📄 Agentic Resume RAG Chatbot")


# --------------------------------------------------
# SESSION STATE INITIALIZATION (CRITICAL)
# --------------------------------------------------
if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None


# --------------------------------------------------
# UPLOAD RESUME
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    os.makedirs("data/resumes", exist_ok=True)
    resume_path = f"data/resumes/{uploaded_file.name}"

    with open(resume_path, "wb") as f:
        f.write(uploaded_file.read())

    resume_text = extract_resume_text(resume_path)
    chunks = chunk_by_section(resume_text)

    embeddings = st.session_state.embedder.embed(chunks)

    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    st.session_state.vector_store = vector_store

    init_tools(st.session_state.embedder, vector_store)
    st.session_state.agent = create_resume_agent()

    st.success("✅ Resume processed successfully!")


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------
if st.session_state.agent:
    question = st.text_input("Ask a question about the resume")

    if question:
        try:
            result = st.session_state.agent.run(question)

            st.subheader("Answer")
            st.write(result.content)

        except Exception as e:
            st.error("❌ Failed to generate answer")
            st.exception(e)
