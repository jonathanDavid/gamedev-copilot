"""Web documentation discovery — subject name → official docs URLs.

Key-less by design (same philosophy as the YouTube tool): a DuckDuckGo HTML
search finds candidate sites, a scoring heuristic picks the one that looks
like official documentation, and a shallow same-host crawl of that root
collects a bounded set of pages to index. Every network touch goes through
an injectable `fetch` callable, so tests run against fixture HTML.
"""
from __future__ import annotations

import re
from typing import Callable
from urllib.parse import unquote, urljoin, urlsplit

Fetch = Callable[[str], str]

_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"

# looks-like-documentation signals in a URL
_DOCSY = re.compile(r"docs?\.|/docs?(/|$)|documentation|manual|reference|readthedocs|/learn(/|$)|/guide", re.I)
# places that are never the official docs
_NOISE = re.compile(r"youtube\.|reddit\.|stackoverflow\.|twitter\.|x\.com|facebook\.|medium\.|wikipedia\.", re.I)
_BINARY = re.compile(r"\.(png|jpe?g|gif|svg|css|js|zip|pdf|ico|woff2?|json|xml|rss|txt)($|\?)", re.I)


def search_docs_urls(subject: str, fetch: Fetch) -> list[str]:
    """Result URLs for '<subject> documentation', in ranking order."""
    html = fetch(_SEARCH_URL.format(query=(subject + " documentation").replace(" ", "+")))
    # DDG wraps results as //duckduckgo.com/l/?uddg=<urlencoded>&rut=…
    urls = [unquote(m.group(1)) for m in re.finditer(r"uddg=([^&\"']+)", html)]
    if not urls:  # markup drift fallback: any absolute link in the page
        urls = re.findall(r"href=\"(https?://[^\"]+)\"", html)
    seen: set[str] = set()
    out = []
    for u in urls:
        u = u.split("&rut=")[0]
        if u not in seen and not u.startswith("https://duckduckgo.com"):
            seen.add(u)
            out.append(u)
    return out


def pick_docs_root(urls: list[str], subject: str) -> str | None:
    """The URL that most looks like the subject's official documentation."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", subject.lower()) if len(t) >= 3]
    best, best_score = None, -999
    for rank, url in enumerate(urls):
        low = url.lower()
        score = -rank  # earlier search results win ties
        if _NOISE.search(low):
            score -= 20
        if _DOCSY.search(low):
            score += 8
        host = urlsplit(low).netloc
        score += sum(4 for t in tokens if t in host)
        score += sum(1 for t in tokens if t in low)
        if score > best_score:
            best, best_score = url, score
    return best


def crawl_docs(root: str, fetch: Fetch, max_pages: int = 12) -> list[str]:
    """The root page plus same-host pages it links to, capped, in order."""
    pages = [root]
    try:
        html = fetch(root)
    except Exception:  # noqa: BLE001 — an unreachable root yields just itself
        return pages
    base = urlsplit(root)
    seen = {root.rstrip("/")}
    for href in re.findall(r"href=[\"']([^\"'#]+)[\"']", html):
        url = urljoin(root, href).split("#")[0]
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.netloc != base.netloc:
            continue
        if _BINARY.search(url):
            continue
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        pages.append(url)
        if len(pages) >= max_pages:
            break
    return pages


def derive_keywords(subject: str, urls: list[str], limit: int = 12) -> list[str]:
    """Router-fallback keywords from the subject plus crawled path segments."""
    words = [t for t in re.findall(r"[a-z0-9]+", subject.lower()) if len(t) >= 3]
    for url in urls:
        tail = urlsplit(url).path.rsplit("/", 1)[-1]
        tail = re.sub(r"\.\w+$", "", tail)
        for w in re.split(r"[-_.]", tail.lower()):
            if len(w) >= 4 and w.isalpha() and w not in words:
                words.append(w)
    return words[:limit]
