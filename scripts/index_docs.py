"""Indexing — the one-time prep step (see GLOSSARY in copilot/rag/chunking.py).

Fetches the real Phaser 3 pages listed in corpus/urls.txt, strips HTML to
text, chunks, embeds with Ollama's nomic-embed-text, and persists to Chroma
(./chroma_db). Re-runnable: existing ids are upserted by source#position.

Usage:  python scripts/index_docs.py            (needs Ollama running)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.domain import load_profile  # noqa: E402
from copilot.llm.ollama_client import OllamaEmbedder, ollama_available  # noqa: E402
from copilot.rag.chunking import chunk_text  # noqa: E402
from copilot.rag.store import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<(p|div|li|h[1-6]|tr|section|article|pre)[^>]*>", "\n\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def main() -> None:
    if not ollama_available():
        sys.exit("Ollama is not running (needed for embeddings). Start it and retry.")
    # Same subject selection as the chat: positional profile path or
    # COPILOT_DOMAIN. Each subject indexes into its own directory.
    profile_arg = sys.argv[1] if len(sys.argv) > 1 else None
    domain = load_profile(profile_arg)
    print(f"indexing subject: {domain.name} → {domain.index_dir}/")
    urls = [
        u.strip()
        for u in (ROOT / domain.urls_file).read_text(encoding="utf-8").splitlines()
        if u.strip() and not u.startswith("#")
    ]
    retriever = Retriever(OllamaEmbedder(), persist_dir=str(ROOT / domain.index_dir))
    total = 0
    for url in urls:
        try:
            r = httpx.get(url, timeout=30.0, follow_redirects=True, headers={"User-Agent": "research-copilot-indexer/1.0"})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"  skip {url} ({e})")
            continue
        text = html_to_text(r.text)
        chunks = chunk_text(text, source=url)
        added = retriever.add(chunks)
        total += added
        print(f"  {url} -> {added} chunks")
    print(f"indexed {total} chunks · collection now holds {retriever.count()}")


if __name__ == "__main__":
    main()
