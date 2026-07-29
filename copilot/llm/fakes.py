"""Deterministic fakes — how the agent stays testable without any model.

A test scripts the FakeLLM with canned responses, runs the whole graph, and
asserts on the path taken and the answer produced. No GPU, no network, no
flakiness. The HashEmbedder gives *stable* vectors where texts sharing words
land near each other — enough to exercise real Chroma retrieval hermetically.
"""
from __future__ import annotations

import hashlib
import math
import re


class FakeLLM:
    """Returns scripted responses in order (or a mapping by substring)."""

    def __init__(self, responses: list[str] | None = None, by_contains: dict[str, str] | None = None):
        self.responses = list(responses or [])
        self.by_contains = by_contains or {}
        self.calls: list[dict] = []  # tests inspect what the agent asked

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        for needle, response in self.by_contains.items():
            if needle in prompt or needle in system:
                return response
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("FakeLLM ran out of scripted responses")


class HashEmbedder:
    """Deterministic bag-of-words hashing into a small vector space.

    Not a real embedding model — but shared tokens produce shared dimensions,
    so cosine similarity behaves sensibly for tests: 'tilemap collision docs'
    is closer to 'how do tilemap collisions work' than to 'sprite animation'.
    """

    # stopwords + short tokens carry no meaning but dominate small texts —
    # filtering them makes content words (tilemap, collision) decide similarity
    _STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
             "how", "do", "i", "is", "are", "my", "it", "that", "this", "between", "as"}

    def __init__(self, dims: int = 128):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dims
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if token in self._STOP or len(token) < 3:
                    continue
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                v[h % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out
