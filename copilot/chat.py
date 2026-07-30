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

import itertools
import sys
import threading
import time
from pathlib import Path

from copilot.domain import load_profile
from copilot.graph import Copilot
from copilot.llm.ollama_client import OllamaEmbedder, OllamaLLM, ollama_available
from copilot.memory.memory import ConversationBuffer, ProjectMemory
from copilot.rag.store import Retriever
from copilot.tools.youtube import YouTubeSearch

ROOT = Path(__file__).resolve().parents[1]

# Friendly live-status text per graph node (shown while the NEXT step runs).
_AFTER_NODE = {
    "route": "routed — working on the answer",
    "retrieve": "docs retrieved — generating (local 7B, be patient)",
    "search_video": "videos found — generating",
    "generate": "answer ready — saving memory",
    "update_memory": "done",
}


class Spinner:
    """Tiny thread that keeps one status line alive while the graph runs.

    Local inference means 60–90 s per docs answer — without this, the CLI
    looks frozen. The graph streams node completions into `bump()`, so the
    line shows real progress, not a fake animation.
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self._label = "thinking"
        self._done: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def bump(self, node: str) -> None:
        self._done.append(node)
        self._label = _AFTER_NODE.get(node, node)

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
        return time.time() - self._t0

    def _run(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while not self._stop.is_set():
            trail = " ".join(f"{n}✓" for n in self._done)
            line = f"\r{next(frames)} {self._label} · {int(time.time() - self._t0)}s  {trail}"
            try:
                sys.stdout.write(line[:100].ljust(100))
            except UnicodeEncodeError:  # exotic consoles: fall back to ASCII
                sys.stdout.write(f"\r* {self._label} {int(time.time() - self._t0)}s".ljust(60))
            sys.stdout.flush()
            self._stop.wait(0.12)


def main() -> None:
    if not ollama_available():
        raise SystemExit(
            "Ollama isn't running on http://localhost:11434.\n"
            "Install: https://ollama.com  ·  then: ollama pull mistral && ollama pull nomic-embed-text"
        )
    # The SUBJECT is chosen at launch, not typed in chat: pass a profile path
    # (chat.sh profiles/fastapi.json) or set COPILOT_DOMAIN. No profile → the
    # game-dev demo. Each subject keeps its own vector index.
    profile_arg = sys.argv[1] if len(sys.argv) > 1 else None
    domain = load_profile(profile_arg)
    retriever = Retriever(OllamaEmbedder(), persist_dir=str(ROOT / domain.index_dir))
    if retriever.count() == 0:
        print(f"⚠ the '{domain.name}' docs index is empty — run scripts/index_docs.py "
              "with the same profile first (docs questions will degrade).")

    project = ProjectMemory(ROOT / "project_memory.json")
    bot = Copilot(
        llm=OllamaLLM(),
        embedder=OllamaEmbedder(),
        retriever=retriever,
        video_search=YouTubeSearch(),
        buffer=ConversationBuffer(max_turns=8),
        project=project,
        domain=domain,
    )

    print(domain.banner)
    print(f"   subject: {domain.name} — switch with `chat profiles/<name>.json` "
          "(see profiles/fastapi.json) or COPILOT_DOMAIN")
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

        spinner = Spinner()
        spinner.start()
        try:
            state = bot.ask(q, on_node=spinner.bump)
        finally:
            elapsed = spinner.stop()
        print(f"\n[path: {' → '.join(state['path'])}]  [label: {state['label']}]  [{elapsed:.0f}s]")
        for i, h in enumerate(state.get("hits", []) or []):
            print(f"  [{i+1}] {h.source}  (score {h.score:.2f})")
        print(f"\ncopilot> {state['answer']}")


if __name__ == "__main__":
    main()
