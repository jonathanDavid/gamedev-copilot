"""LLM + embedding interfaces — the testability spine.

GLOSSARY (Inference): "inference" is just *running the model to produce
output*. "Local inference" means that computation happens on YOUR machine
(Ollama serving Mistral) instead of a cloud API.

Every place the agent needs a model goes through these two small interfaces.
That single decision is what makes the whole project testable: tests plug in
deterministic fakes (see `copilot/llm/fakes.py`), production plugs in Ollama
(`copilot/llm/ollama_client.py`). Nothing else in the codebase knows or cares
which one it's talking to.
"""
from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    """Anything that can complete a prompt into text."""

    def complete(self, system: str, prompt: str) -> str:
        """One inference call: system instructions + user prompt -> text."""
        ...


class Embedder(Protocol):
    """GLOSSARY (Embeddings): turns text into a vector — a list of numbers
    that captures *meaning*, so "tilemap collision" and "tiles blocking the
    player" land near each other even though the words differ. Vectors are
    what make search-by-meaning possible."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
