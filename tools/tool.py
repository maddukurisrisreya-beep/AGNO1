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
    if not _embedder or not _vector_store:
        return ""

    try:
        query_embedding = _embedder.embed([query])
        results = _vector_store.search(query_embedding, k=3)
        return " ".join(results)
    except Exception:
        return ""
