"""Ollama client — real local inference.

GLOSSARY (Ollama / Mistral): Ollama is an app that downloads open models and
serves them on your machine behind a simple HTTP API (default
http://localhost:11434). Mistral is one such model — 7B parameters, small
enough for a decent laptop. Slow-ish, free, private: no tokens, no cloud.
"""
from __future__ import annotations

import httpx

DEFAULT_BASE = "http://localhost:11434"


class OllamaLLM:
    def __init__(self, model: str = "mistral", base_url: str = DEFAULT_BASE, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        r = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                # low temperature: the router especially must be boring and reliable
                "options": {"temperature": 0.2},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


class OllamaEmbedder:
    """nomic-embed-text: a small local embedding model served by Ollama."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = DEFAULT_BASE, timeout: float = 60.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            r = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=self.timeout,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out


def ollama_available(base_url: str = DEFAULT_BASE) -> bool:
    """Used by the CLI to fail with a friendly message, and by integration
    tests to skip themselves on machines without Ollama."""
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=2.0).status_code == 200
    except Exception:
        return False
