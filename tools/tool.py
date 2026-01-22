from agno.tools import tool

# These will be injected from app.py
_embedder = None
_vector_store = None


def init_tools(embedder, vector_store):
    """
    Initialize shared objects for tools
    """
    global _embedder, _vector_store
    _embedder = embedder
    _vector_store = vector_store


@tool
def retrieve_resume_context(query: str) -> str:
    """
    Retrieve relevant resume content for a given query.
    Always returns actual resume text from the vector store.
    """

    if _embedder is None or _vector_store is None:
        return "Resume data is not initialized."

    # Embed the query
    query_embedding = _embedder.embed([query])

    # Search vector store
    results = _vector_store.search(query_embedding, top_k=3)

    if not results:
        return "No relevant information found in the resume."

    # Combine retrieved chunks
    context = "\n".join(results)
    return context
