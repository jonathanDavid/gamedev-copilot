"""Shared indexing pipeline: URL list → text → chunks → embeddings → Chroma.

Used by scripts/index_docs.py (index a profile's url list) and
scripts/new_subject.py (discover docs on the web, then index them).
"""
from __future__ import annotations

import re
from typing import Callable

import httpx

from copilot.rag.chunking import chunk_text
from copilot.rag.store import Retriever

USER_AGENT = "research-copilot-indexer/1.0"


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<(p|div|li|h[1-6]|tr|section|article|pre)[^>]*>", "\n\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def fetch_page(url: str) -> str:
    r = httpx.get(url, timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text


def index_urls(urls: list[str], retriever: Retriever, log: Callable[[str], None] = print) -> int:
    """Fetch, chunk, embed and store each page. Returns chunks added."""
    total = 0
    for url in urls:
        try:
            html = fetch_page(url)
        except Exception as e:  # noqa: BLE001 — a dead page must not kill the run
            log(f"  skip {url} ({e})")
            continue
        chunks = chunk_text(html_to_text(html), source=url)
        added = retriever.add(chunks)
        total += added
        log(f"  {url} -> {added} chunks")
    return total
