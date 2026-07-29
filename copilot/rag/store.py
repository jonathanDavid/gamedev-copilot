"""Vector store — Chroma.

GLOSSARY (Chroma / vector database): a database built to store embeddings and
answer "which stored chunks are closest in meaning to this query?" in
milliseconds. Runs locally as a Python library — persistent on disk for the
real corpus, in-memory (ephemeral) for tests.

GLOSSARY (Retriever): the glue — takes your question → embeds it → asks
Chroma for the top matches → returns those chunks. That's `Retriever.search`.

GLOSSARY (Hallucination): a model confidently inventing facts (a Phaser
method that doesn't exist). RAG reduces it by grounding the answer in real
retrieved text — the generate step is told to answer FROM these chunks and
cite them, instead of from vibes.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import chromadb

from copilot.llm.base import Embedder
from copilot.rag.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    score: float  # cosine similarity, higher = closer in meaning


class Retriever:
    def __init__(self, embedder: Embedder, persist_dir: str | None = None, collection: str | None = None):
        # Ephemeral (RAM) for tests; persistent (disk) for the real corpus.
        # NOTE: Chroma's ephemeral clients can share in-process state, so
        # ephemeral retrievers default to a UNIQUE collection name — otherwise
        # two "independent" test retrievers silently see each other's chunks.
        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        )
        if collection is None:
            collection = "phaser_docs" if persist_dir else f"ephemeral_{uuid4().hex[:12]}"
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._col.add(
            ids=[f"{c.source}#{c.position}" for c in chunks],
            embeddings=self._embedder.embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "position": c.position} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self._col.query(query_embeddings=self._embedder.embed([query]), n_results=min(k, self.count()))
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(text=doc, source=str(meta["source"]), score=1.0 - dist))
        return hits

    def count(self) -> int:
        return self._col.count()
