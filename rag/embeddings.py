import os

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 16):
            response = httpx.post("http://127.0.0.1:11434/api/embed",
                json={"model": self.model, "input": texts[start:start + 16]}, timeout=300)
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embeddings():
    """Ollama local embeddings by default; Hugging Face is an optional fallback."""
    backend = os.getenv("POLICY_EMBEDDING_BACKEND", "ollama" if os.getenv("LLM_BACKEND") == "ollama" else "huggingface")
    if backend == "ollama":
        return OllamaEmbeddings(os.getenv("POLICY_EMBEDDING_MODEL", "qwen3-embedding:0.6b"))
    from langchain_huggingface import HuggingFaceEmbeddings
    model = os.getenv("POLICY_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})
