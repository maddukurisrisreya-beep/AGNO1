from agno.tools import tool

_embedder = None
_vector_store = None


def init_tools(embedder, vector_store):
    global _embedder, _vector_store
    _embedder = embedder
    _vector_store = vector_store


@tool
def retrieve_resume_context(query: str) -> str:
    """
    Retrieve relevant resume content for a query.
    """

    if _embedder is None or _vector_store is None:
        return "Resume not loaded. Please upload your resume again."

    query_embedding = _embedder.embed([query])
    results = _vector_store.search(query_embedding, k=3)

    if not results:
        return "No relevant information found in the resume."

    return "\n".join(results)
