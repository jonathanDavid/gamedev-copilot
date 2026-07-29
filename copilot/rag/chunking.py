"""Chunking — splitting documents before indexing.

GLOSSARY (Indexing): the one-time prep step — split documents into pieces and
store them in searchable form BEFORE any question is asked.

GLOSSARY (Chunks): the pieces themselves. Models retrieve and reason better
over small focused passages (~300–500 words) than over whole files, so we cut
on paragraph boundaries with a little overlap to avoid slicing a sentence's
meaning in half at a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # URL or file the chunk came from — powers citations
    position: int  # chunk index within its source


def chunk_text(text: str, source: str, max_words: int = 400, overlap_words: int = 60) -> list[Chunk]:
    """Greedy paragraph packing with word-overlap between consecutive chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal current, count
        if not current:
            return
        chunks.append(Chunk(text="\n\n".join(current), source=source, position=len(chunks)))
        # keep the tail as overlap so boundary sentences appear in both chunks
        tail_words = " ".join(current).split()[-overlap_words:]
        current = [" ".join(tail_words)] if tail_words else []
        count = len(tail_words)

    for para in paragraphs:
        words = len(para.split())
        if count + words > max_words and current:
            flush()
        current.append(para)
        count += words
    flush()
    # drop a trailing overlap-only chunk (it duplicates the previous tail)
    if len(chunks) > 1 and len(chunks[-1].text.split()) <= overlap_words:
        chunks.pop()
    return chunks
