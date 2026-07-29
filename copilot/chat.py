"""CLI REPL — the whole agent in a terminal.

Run:  python -m copilot.chat            (needs Ollama running + indexed docs)

Every answer prints the GRAPH PATH the question actually took (route →
retrieve → generate → update_memory), the router's label, and doc citations —
so you can watch the architecture work, not just read about it.

Commands:
  /remember <fact>   save a durable project decision (long-term memory)
  /facts             list saved decisions
  /forget <n>        drop decision n
  /quit
"""
from __future__ import annotations

from pathlib import Path

from copilot.graph import Copilot
from copilot.llm.ollama_client import OllamaEmbedder, OllamaLLM, ollama_available
from copilot.memory.memory import ConversationBuffer, ProjectMemory
from copilot.rag.store import Retriever
from copilot.tools.youtube import YouTubeSearch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if not ollama_available():
        raise SystemExit(
            "Ollama isn't running on http://localhost:11434.\n"
            "Install: https://ollama.com  ·  then: ollama pull mistral && ollama pull nomic-embed-text"
        )
    retriever = Retriever(OllamaEmbedder(), persist_dir=str(ROOT / "chroma_db"))
    if retriever.count() == 0:
        print("⚠ docs index is empty — run scripts/index_docs.py first (docs questions will degrade).")

    project = ProjectMemory(ROOT / "project_memory.json")
    bot = Copilot(
        llm=OllamaLLM(),
        embedder=OllamaEmbedder(),
        retriever=retriever,
        video_search=YouTubeSearch(),
        buffer=ConversationBuffer(max_turns=8),
        project=project,
    )

    print("🎮 gamedev-copilot — Phaser · Godot · PixiJS · Ebitengine · Bevy + 7 more (/quit to exit)")
    if project.facts():
        print("   project memory:", "; ".join(project.facts()))
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q.startswith("/remember "):
            project.remember(q[len("/remember "):])
            print("saved.")
            continue
        if q == "/facts":
            for i, f in enumerate(project.facts()):
                print(f"  [{i}] {f}")
            continue
        if q.startswith("/forget "):
            removed = project.forget(int(q.split()[1]))
            print(f"forgot: {removed}" if removed else "no such fact")
            continue

        state = bot.ask(q)
        print(f"\n[path: {' → '.join(state['path'])}]  [label: {state['label']}]")
        for i, h in enumerate(state.get("hits", []) or []):
            print(f"  [{i+1}] {h.source}  (score {h.score:.2f})")
        print(f"\ncopilot> {state['answer']}")


if __name__ == "__main__":
    main()
