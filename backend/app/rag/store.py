"""Policy document ingestion and retrieval (Chroma, local)."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

COLLECTION = "policy_documents"
_HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str  # file name, e.g. "sample_policy.md"
    section: str  # e.g. "Section 1: Home Water Damage Coverage"
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.section}" if self.section else self.source


def load_policy_chunks(docs_dir: Path) -> list[Document]:
    """Split every markdown file into section-level chunks with citation metadata."""
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    # Long sections are split further; short ones (like the sample) stay whole.
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    chunks: list[Document] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        for doc in size_splitter.split_documents(header_splitter.split_text(path.read_text("utf-8"))):
            section = doc.metadata.get("subsection") or doc.metadata.get("section") or ""
            if not section:
                continue  # skip the bare document title chunk
            doc.metadata = {
                "source": path.name,
                "title": doc.metadata.get("title", ""),
                "section": section,
            }
            chunks.append(doc)
    return chunks


class PolicyStore:
    def __init__(self, embeddings: Embeddings, persist_dir: Path | None = None):
        self._db = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(persist_dir) if persist_dir else None,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, docs_dir: Path) -> int:
        """Idempotently (re)index the policy documents. Returns the chunk count."""
        chunks = load_policy_chunks(docs_dir)
        ids = [
            hashlib.sha1(f"{c.metadata['source']}|{c.metadata['section']}|{c.page_content}".encode()).hexdigest()
            for c in chunks
        ]
        existing = self._db.get(include=[])["ids"]
        if existing:
            self._db.delete(ids=existing)
        if chunks:
            self._db.add_documents(chunks, ids=ids)
        return len(chunks)

    def search(self, query: str, k: int = 3, min_score: float = 0.0, max_gap: float = 1.0) -> list[RetrievedChunk]:
        """Top-k chunks, dropping any below `min_score` or more than `max_gap` below the best hit.

        The absolute floor removes off-topic matches; the relative gap removes sections that are
        merely "less wrong" than the real answer, while keeping several when they score alike.
        """
        results = self._db.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return []
        floor = max(min_score, max(score for _, score in results) - max_gap)
        return [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
                section=doc.metadata.get("section", ""),
                score=round(score, 4),
            )
            for doc, score in results
            if score >= floor
        ]
