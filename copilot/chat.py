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

from copilot.domain import GAMEDEV, load_profile
from copilot.graph import Copilot
from copilot.llm.ollama_client import OllamaEmbedder, OllamaLLM, ollama_available
from copilot.memory.memory import ConversationBuffer, ProjectMemory
from copilot.rag.store import Retriever
from copilot.subject import ensure_subject, extract_subject_request
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
    # Piped/redirected stdout on Windows defaults to cp1252, which cannot
    # encode the banner emoji — never crash over a glyph.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not ollama_available():
        raise SystemExit(
            "Ollama isn't running on http://localhost:11434.\n"
            "Install: https://ollama.com  ·  then: ollama pull mistral && ollama pull nomic-embed-text"
        )
    project = ProjectMemory(ROOT / "project_memory.json")

    def build_bot(domain):
        retriever = Retriever(OllamaEmbedder(), persist_dir=str(ROOT / domain.index_dir))
        if retriever.count() == 0:
            print(f"⚠ the '{domain.name}' docs index is empty — docs questions will degrade.")
        return Copilot(
            llm=OllamaLLM(),
            embedder=OllamaEmbedder(),
            retriever=retriever,
            video_search=YouTubeSearch(),
            buffer=ConversationBuffer(max_turns=8),
            project=project,
            domain=domain,
        )

    def become_expert(subject: str):
        """Discover + index the subject's docs on the web, then reskin."""
        report = ensure_subject(subject, ROOT, OllamaEmbedder())
        if report.created:
            print(f"✓ now an expert on {report.profile.name} — indexed {report.chunks} chunks "
                  f"from {report.pages} pages at {report.root}")
        else:
            print(f"✓ switched to existing subject: {report.profile.name}")
        return build_bot(report.profile)

    # The SUBJECT: a profile path argument / COPILOT_DOMAIN wins; otherwise
    # the chat ASKS — type anything ("Unity 3D", "Rust") and an agent finds
    # and indexes the docs from the internet. Enter keeps the demo subject.
    profile_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if profile_arg or not sys.stdin.isatty():
        domain = load_profile(profile_arg)
        bot = build_bot(domain)
    else:
        answer = input("What should I be an expert on? (Enter = 2D game development demo) ").strip()
        if answer:
            bot = become_expert(answer)
            domain = bot.domain
        else:
            domain = GAMEDEV
            bot = build_bot(domain)

    print(domain.banner)
    print("   switch anytime: say \"I want you to be an expert in <subject>\" "
          "or /subject <subject>")
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
        if q.startswith("/subject "):
            bot = become_expert(q[len("/subject "):].strip())
            continue
        # "I want you to be an expert in X …" — switch subjects mid-chat
        subject_request = extract_subject_request(q)
        if subject_request:
            bot = become_expert(subject_request)
            print("   ask your question again — it will be answered from the new corpus.")
            continue

        if sys.stdout.isatty():
            spinner = Spinner()
            spinner.start()
            try:
                state = bot.ask(q, on_node=spinner.bump)
            finally:
                elapsed = spinner.stop()
        else:  # piped/scripted runs: no spinner control characters in output
            t0 = time.time()
            state = bot.ask(q)
            elapsed = time.time() - t0
        print(f"\n[path: {' → '.join(state['path'])}]  [label: {state['label']}]  [{elapsed:.0f}s]")
        for i, h in enumerate(state.get("hits", []) or []):
            print(f"  [{i+1}] {h.source}  (score {h.score:.2f})")
        print(f"\ncopilot> {state['answer']}")


if __name__ == "__main__":
    main()
