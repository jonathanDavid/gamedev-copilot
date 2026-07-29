"""Memory — two layers with two different lifetimes.

GLOSSARY (Conversation buffer / short-term memory): models remember NOTHING
between calls — "memory" is literally re-sending recent chat history with
every prompt. The buffer keeps the last N turns so "and how do I make that
jump feel better?" resolves against the previous answer.

GLOSSARY (Long-term memory / context management): deciding what survives
beyond the session. The context window (the model's input size limit) can't
hold everything forever, so durable FACTS — "project uses Phaser 3 +
TypeScript, arcade physics" — are stored in a small JSON file and injected
into every prompt, while old chit-chat is simply dropped (oldest turns fall
off the buffer). That triage IS context management.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    text: str


class ConversationBuffer:
    def __init__(self, max_turns: int = 8):
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, text: str) -> None:
        self._turns.append(Turn(role, text))

    def render(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self._turns)

    def __len__(self) -> int:
        return len(self._turns)


class ProjectMemory:
    """Durable project decisions, persisted as JSON on disk."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._facts: list[str] = []
        if self._path.exists():
            self._facts = json.loads(self._path.read_text(encoding="utf-8"))

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self._facts:
            self._facts.append(fact)
            self._save()

    def forget(self, index: int) -> str | None:
        if 0 <= index < len(self._facts):
            fact = self._facts.pop(index)
            self._save()
            return fact
        return None

    def facts(self) -> list[str]:
        return list(self._facts)

    def render(self) -> str:
        if not self._facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self._facts)
        return f"Known project decisions (always respect these):\n{lines}"

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._facts, indent=2, ensure_ascii=False), encoding="utf-8")
