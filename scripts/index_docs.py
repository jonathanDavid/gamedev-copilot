"""Indexing — the one-time prep step (see GLOSSARY in copilot/rag/chunking.py).

Fetches the profile's documentation pages, strips HTML to text, chunks,
embeds with Ollama's nomic-embed-text, and persists to the profile's own
Chroma directory. Re-runnable: existing ids are upserted by source#position.

Usage:  python scripts/index_docs.py [profiles/<name>.json]
        (or COPILOT_DOMAIN=profiles/<name>.json; default = game-dev demo)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.domain import load_profile  # noqa: E402
from copilot.indexing import index_urls  # noqa: E402
from copilot.llm.ollama_client import OllamaEmbedder, ollama_available  # noqa: E402
from copilot.rag.store import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
    total = index_urls(urls, retriever)
    print(f"indexed {total} chunks · collection now holds {retriever.count()}")


if __name__ == "__main__":
    main()
