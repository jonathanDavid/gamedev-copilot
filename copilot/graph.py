"""The agent graph — explicit nodes and edges, no freeform loop.

GLOSSARY (Graph / nodes / edges): LangGraph models the workflow as a graph.
Nodes = steps (route, retrieve, search_video, generate, update_memory).
Edges = allowed transitions, including the conditional branch out of the
router. Versus a freeform agent that decides everything on the fly, the flow
is controlled, visible, and debuggable — `state["path"]` records exactly
which nodes ran for every question, and the tests assert on it.

    route ──docs──► retrieve ─────┐
      │──video──► search_video ───┼──► generate ──► update_memory ──► END
      └──chat──────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from copilot.domain import GAMEDEV, DomainProfile
from copilot.llm.base import LLM, Embedder
from copilot.memory.memory import ConversationBuffer, ProjectMemory
from copilot.nodes import router as router_node
from copilot.rag.store import Hit, Retriever
from copilot.tools.youtube import Video, VideoSearch


class AgentState(TypedDict, total=False):
    question: str
    label: str                 # router's decision: docs | video | chat
    hits: list[Hit]            # retrieved chunks (docs path)
    videos: list[Video]        # tool results (video path)
    answer: str
    path: list[str]            # trace of nodes that actually ran


@dataclass
class Copilot:
    """Wires the graph once; `ask()` runs one question through it."""

    llm: LLM
    embedder: Embedder
    retriever: Retriever
    video_search: VideoSearch
    buffer: ConversationBuffer = field(default_factory=ConversationBuffer)
    project: ProjectMemory | None = None
    # The subject is CONFIGURATION, not architecture — swap this profile and
    # the same graph researches any topic (see copilot/domain.py).
    domain: DomainProfile = field(default_factory=lambda: GAMEDEV)

    # Cosine score below which retrieval is treated as off-topic. Calibrated
    # for the REAL embedder (nomic-embed-text): in-corpus questions score
    # ~0.66-0.75, out-of-domain ones ~0.60-0.62. The hash embedder used in
    # tests has a different scale — tests pass their own threshold.
    low_relevance: float = 0.65

    def __post_init__(self) -> None:
        g = StateGraph(AgentState)
        g.add_node("route", self._route)
        g.add_node("retrieve", self._retrieve)
        g.add_node("search_video", self._search_video)
        g.add_node("generate", self._generate)
        g.add_node("update_memory", self._update_memory)

        g.set_entry_point("route")
        g.add_conditional_edges(
            "route",
            lambda s: s["label"],
            {"docs": "retrieve", "video": "search_video", "chat": "generate"},
        )
        g.add_edge("retrieve", "generate")
        g.add_edge("search_video", "generate")
        g.add_edge("update_memory", END)
        g.add_edge("generate", "update_memory")
        self._graph = g.compile()

    # ---- nodes ----------------------------------------------------------

    def _route(self, state: AgentState) -> AgentState:
        label = router_node.route(self.llm, state["question"], self.domain.docs_keywords)
        return {"label": label, "path": state.get("path", []) + ["route"]}

    def _retrieve(self, state: AgentState) -> AgentState:
        hits = self.retriever.search(state["question"], k=4)
        return {"hits": hits, "path": state["path"] + ["retrieve"]}

    def _search_video(self, state: AgentState) -> AgentState:
        videos = self.video_search.search(state["question"], k=3)
        return {"videos": videos, "path": state["path"] + ["search_video"]}

    def _generate(self, state: AgentState) -> AgentState:
        system, prompt = self._build_prompt(state)
        answer = self.llm.complete(system, prompt)
        return {"answer": answer, "path": state["path"] + ["generate"]}

    def _update_memory(self, state: AgentState) -> AgentState:
        self.buffer.add("user", state["question"])
        self.buffer.add("assistant", state["answer"])
        return {"path": state["path"] + ["update_memory"]}

    # ---- prompt assembly -------------------------------------------------

    def _build_prompt(self, state: AgentState) -> tuple[str, str]:
        parts: list[str] = []
        facts = self.project.render() if self.project else ""
        if facts:
            parts.append(facts)
        history = self.buffer.render()
        if history:
            parts.append(f"Recent conversation:\n{history}")

        label = state["label"]
        if label == "docs":
            hits = state.get("hits", [])
            if hits:
                ctx = "\n\n".join(f"[{i+1}] (source: {h.source})\n{h.text}" for i, h in enumerate(hits))
                parts.append(
                    "Documentation excerpts (answer FROM these; cite like [1]; "
                    "if they don't cover it, say so honestly rather than inventing APIs):\n" + ctx
                )
                # Weak retrieval = the index probably doesn't cover the topic
                # (e.g. asking a game-dev index about Unity). A 7B left alone
                # will summarize the wrong excerpts instead of admitting it —
                # measured live — so the admission gets demanded explicitly.
                if max(h.score for h in hits) < self.low_relevance:
                    parts.append(
                        "WARNING: every excerpt above matched this question only weakly. "
                        "If they do not actually answer it, START your reply by saying the "
                        "indexed documentation does not cover this topic — do not summarize "
                        "unrelated excerpts."
                    )
            else:
                parts.append(
                    "No documentation matched. Say the docs index has no coverage and answer "
                    "only from general knowledge, clearly labeled as such."
                )
            system = (
                f"You are {self.domain.persona}. "
                "Ground every API claim in the provided excerpts with [n] citations."
            )
        elif label == "video":
            videos = state.get("videos", [])
            if videos:
                vlist = "\n".join(f"- {v.title} — {v.channel} ({v.duration}) {v.url}" for v in videos)
                parts.append(f"Tutorial videos found:\n{vlist}")
                system = (
                    f"You are {self.domain.persona}. Recommend the most relevant of the "
                    "found videos (with their URLs) and say in one line what each covers."
                )
            else:
                parts.append("The video search returned nothing.")
                system = f"You are {self.domain.persona}. Apologize that no videos were found and answer briefly yourself."
        else:
            system = f"You are {self.domain.persona}. Be friendly and concise."

        parts.append(f"User question: {state['question']}")
        return system, "\n\n".join(parts)

    # ---- public API ------------------------------------------------------

    def ask(self, question: str, on_node=None) -> AgentState:
        """Run one question through the graph.

        `on_node`, if given, is called with each node's name as it COMPLETES —
        LangGraph streams state after every node, which is what lets the CLI
        show live progress instead of 60–90 s of silence during local
        inference.
        """
        state: AgentState = {"question": question, "path": []}
        if on_node is None:
            return self._graph.invoke(state)
        last: AgentState = state
        seen = 0
        for snapshot in self._graph.stream(state, stream_mode="values"):
            last = snapshot
            path = snapshot.get("path", [])
            while seen < len(path):
                on_node(path[seen])
                seen += 1
        return last
