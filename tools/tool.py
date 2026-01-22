from agno.tools import tool
from embeddings.embedder import Embedder
from vectorstore.store import VectorStore

_embedder = None
_vector_store = None


def init_tools(embedder: Embedder, vector_store: VectorStore):
    global _embedder, _vector_store
    _embedder = embedder
    _vector_store = vector_store


@tool
def retrieve_resume_context(query: str) -> str:
    query_embedding = _embedder.embed(query)
    results = _vector_store.search(query_embedding, k=3)

    if not results:
        return "Not mentioned in the resume."

    return " ".join(results)
