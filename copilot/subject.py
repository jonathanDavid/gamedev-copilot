"""Subject acquisition — "be an expert on X" becomes a real, indexed corpus.

The pipeline the chat runs when the user names a subject:

    search the web for "<subject> documentation"   (tools/webdocs, key-less)
    → pick the site that looks like the official docs
    → shallow same-host crawl (bounded)
    → write profiles/<slug>.json + profiles/<slug>-urls.txt
    → fetch/chunk/embed the pages into the subject's own Chroma dir

Everything network- or model-touching is injectable, so the whole pipeline
is hermetically testable with fixture HTML and the hash embedder.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from copilot.domain import DomainProfile, load_profile
from copilot.indexing import fetch_page, index_urls
from copilot.llm.base import Embedder
from copilot.rag.store import Retriever
from copilot.tools.webdocs import crawl_docs, derive_keywords, pick_docs_root, search_docs_urls

# "make me/be/you're an expert on|in|about X" — English and Spanish
_EXPERT = re.compile(r"\bexpert[oa]?\s+(?:on|in|about|en|de)\s+(.+)", re.I)
_CUTOFF = re.compile(r"\b(?:then|and)\b|[,.;?!]", re.I)


def extract_subject_request(text: str) -> str | None:
    """The X in 'I want you to be an expert on X …', or None.

    Trailing clauses are cut ('…expert on unity 3D then answer me what a
    gameobject is' → 'unity 3D') and the subject is capped at 6 words.
    """
    m = _EXPERT.search(text)
    if not m:
        return None
    subject = _CUTOFF.split(m.group(1))[0].strip().strip("\"'")
    return " ".join(subject.split()[:6]) or None


def slugify(subject: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", subject.lower()))[:60]


@dataclass
class SubjectReport:
    profile: DomainProfile
    created: bool           # False = an existing profile was reused
    pages: int = 0
    chunks: int = 0
    root: str = ""


def ensure_subject(
    subject: str,
    base_dir: Path,
    embedder: Embedder,
    fetch: Callable[[str], str] = fetch_page,
    log: Callable[[str], None] = print,
    max_pages: int = 12,
) -> SubjectReport:
    """Load the subject's profile if it exists; otherwise discover + index it."""
    slug = slugify(subject)
    profile_path = base_dir / "profiles" / f"{slug}.json"
    if profile_path.exists():
        log(f"  using existing profile: {profile_path.name}")
        return SubjectReport(profile=load_profile(str(profile_path)), created=False)

    log(f"  searching the web for {subject!r} documentation…")
    results = search_docs_urls(subject, fetch)
    root = pick_docs_root(results, subject)
    if not root:
        raise RuntimeError(f"found no documentation site for {subject!r}")
    log(f"  official-looking docs: {root}")

    urls = crawl_docs(root, fetch, max_pages=max_pages)
    log(f"  crawled {len(urls)} pages — writing profile + indexing")

    profile = DomainProfile(
        name=subject,
        banner=f"🔬 research-copilot · {subject} (/quit to exit)",
        persona=f"a concise {subject} research copilot",
        docs_keywords=tuple(derive_keywords(subject, urls)),
        urls_file=f"profiles/{slug}-urls.txt",
        index_dir=f"chroma_db_{slug}",
    )
    (base_dir / "profiles").mkdir(exist_ok=True)
    (base_dir / profile.urls_file).write_text(
        f"# Auto-discovered corpus for {subject!r} (root: {root})\n" + "\n".join(urls) + "\n",
        encoding="utf-8",
    )
    profile_path.write_text(
        json.dumps(
            {
                "name": profile.name,
                "banner": profile.banner,
                "persona": profile.persona,
                "docs_keywords": list(profile.docs_keywords),
                "urls_file": profile.urls_file,
                "index_dir": profile.index_dir,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    retriever = Retriever(embedder, persist_dir=str(base_dir / profile.index_dir))
    chunks = index_urls(urls, retriever, log=log)
    return SubjectReport(profile=profile, created=True, pages=len(urls), chunks=chunks, root=root)
