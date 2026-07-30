"""Domain profile — the ONE place a subject lives.

The agent's architecture (router → RAG → video tool → generate → memory)
knows nothing about game development. Everything subject-specific — the
persona in the prompts, the CLI banner, the router's keyword fallback, and
which documentation gets indexed — comes from a DomainProfile.

Point the copilot at ANY subject with a small JSON profile:

    {
      "name": "FastAPI",
      "banner": "🔬 research-copilot · FastAPI (/quit to exit)",
      "persona": "a concise FastAPI research copilot",
      "docs_keywords": ["fastapi", "pydantic", "dependency", "middleware"],
      "urls_file": "corpus/fastapi-urls.txt"
    }

then:  COPILOT_DOMAIN=profiles/fastapi.json python scripts/index_docs.py
       COPILOT_DOMAIN=profiles/fastapi.json python -m copilot.chat

The shipped demo profile is 2D game development (12 frameworks indexed).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainProfile:
    name: str
    banner: str
    persona: str
    docs_keywords: tuple[str, ...] = ()
    urls_file: str = "corpus/urls.txt"
    # Each subject gets its OWN vector store — otherwise indexing a second
    # subject would silently mix corpora inside one Chroma collection.
    index_dir: str = "chroma_db"


GAMEDEV = DomainProfile(
    name="2D game development",
    banner="🎮 research-copilot · 2D game dev — Phaser · Godot · PixiJS · Ebitengine · Bevy + 7 more (/quit to exit)",
    persona=(
        "a concise 2D game-development research copilot (Phaser, PixiJS, Excalibur, "
        "Kaplay, Godot, Pygame, MonoGame, libGDX, Ebitengine, Bevy, raylib)"
    ),
    docs_keywords=(
        "collision", "tilemap", "sprite", "physics", "camera", "scene", "animation",
        "phaser", "pixi", "excalibur", "kaplay", "godot", "pygame", "monogame",
        "libgdx", "ebiten", "bevy", "raylib", "entity", "ecs",
    ),
    urls_file="corpus/urls.txt",
)


def load_profile(path: str | None = None) -> DomainProfile:
    """The demo profile, unless COPILOT_DOMAIN (or `path`) names a JSON one."""
    path = path or os.environ.get("COPILOT_DOMAIN")
    if not path:
        return GAMEDEV
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["docs_keywords"] = tuple(data.get("docs_keywords", ()))
    if "index_dir" not in data:
        slug = "".join(c if c.isalnum() else "-" for c in data["name"].lower()).strip("-")
        data["index_dir"] = f"chroma_db_{slug}"
    return DomainProfile(**data)
