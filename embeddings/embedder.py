from transformers import AutoTokenizer, AutoModel
import torch


class Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        self.model = AutoModel.from_pretrained("intfloat/e5-small-v2")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
