import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []

    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        if len(self.texts) == 0:
            return []

        query_embedding = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        return [
            self.texts[i]
            for i in indices[0]
            if i < len(self.texts)
        ]
