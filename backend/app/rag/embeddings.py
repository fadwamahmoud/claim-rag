"""Embedding backends for the policy vector store."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by does do for from how i if in is it my of on or the to up what with".split()
)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embedder using the hashing trick.

    It captures lexical overlap only, which is enough for offline tests and CI
    where downloading a neural model is not possible.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            token = token.rstrip("s") or token  # crude plural folding
            digest = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX sentence embeddings via `fastembed` (no API key, CPU only)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "hashing":
        return HashingEmbeddings()
    return FastEmbedEmbeddings(settings.embedding_model)
