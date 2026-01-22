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
# LOAD ENV VARIABLES (LOCAL USE)
# --------------------------------------------------
load_dotenv(dotenv_path=".env", override=True)

# HARD CHECK — FAIL EARLY
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not loaded. Check your .env file.")
    st.stop()

# DEBUG (REMOVE AFTER CONFIRMING)
st.write("🔑 KEY CHECK:", repr(os.getenv("GROQ_API_KEY")))


# --------------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------------
st.set_page_config(page_title="Agentic Resume RAG", layout="centered")
st.title("📄 Agentic Resume RAG Chatbot")


# --------------------------------------------------
# INITIALIZE COMPONENTS
# --------------------------------------------------
embedder = Embedder()
vector_store = None
agent = None


# --------------------------------------------------
# UPLOAD RESUME
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    os.makedirs("data/resumes", exist_ok=True)
    resume_path = f"data/resumes/{uploaded_file.name}"

    with open(resume_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract resume text
    resume_text = extract_resume_text(resume_path)

    # Chunk by section
    chunks = chunk_by_section(resume_text)

    # Create embeddings
    embeddings = embedder.embed(chunks)

    # Vector store
    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    # Initialize tools
    init_tools(embedder, vector_store)

    # Create agent
    agent = create_resume_agent()

    st.success("✅ Resume processed successfully!")


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------
if agent:
    question = st.text_input("Ask a question about the resume")

    if question:
        try:
            result = agent.run(question)

            st.subheader("Answer")
            st.write(result.content)

        except Exception as e:
            st.error("❌ Failed to generate answer")
            st.exception(e)
