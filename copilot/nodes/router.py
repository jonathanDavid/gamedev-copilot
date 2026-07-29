"""The router node — classification, nothing else.

GLOSSARY (LangGraph router node): a step whose ONLY job is deciding the path.
The LLM reads the question and outputs a label — "docs", "video", or "chat" —
and the graph branches on that label. It generates no answer here.

Reliability tricks that matter in practice:
- the model is asked for STRICT one-word output (temperature is low too);
- the output is parsed defensively (first known label found wins);
- if the model says something unusable, a keyword heuristic decides instead —
  a router that crashes on weird output is worse than a dumb-but-total one.
"""
from __future__ import annotations

import re

from copilot.llm.base import LLM

LABELS = ("docs", "video", "chat")

_SYSTEM = (
    "You are a router for a game-development copilot. Classify the user's "
    "message into exactly one word:\n"
    "docs  — a technical how/what/why question answerable from API docs or guides\n"
    "video — the user wants a tutorial, walkthrough, or something to watch\n"
    "chat  — greetings, opinions, project decisions, anything else\n"
    "Reply with ONLY the single word."
)

_VIDEO_HINTS = re.compile(r"\b(video|tutorial|watch|youtube|walkthrough|course|paso a paso)\b", re.I)
_DOCS_HINTS = re.compile(r"\b(how|what|why|api|method|function|error|collision|tilemap|sprite|physics|camera|scene|animation|phaser|pixi|excalibur|kaplay|godot|pygame|monogame|libgdx|ebiten|bevy|raylib|node|entity|ecs|c[oó]mo|qu[eé])\b", re.I)


def heuristic_route(question: str) -> str:
    if _VIDEO_HINTS.search(question):
        return "video"
    if _DOCS_HINTS.search(question):
        return "docs"
    return "chat"


def route(llm: LLM, question: str) -> str:
    raw = llm.complete(_SYSTEM, question).lower()
    for label in LABELS:
        if label in raw.split() or raw.strip() == label:
            return label
    # salvage: label mentioned anywhere in a wordy reply
    for label in LABELS:
        if label in raw:
            return label
    return heuristic_route(question)
