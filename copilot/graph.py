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
        label = router_node.route(self.llm, state["question"])
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
            else:
                parts.append(
                    "No documentation matched. Say the docs index has no coverage and answer "
                    "only from general knowledge, clearly labeled as such."
                )
            system = (
                "You are a concise 2D game-development copilot (Phaser-focused). "
                "Ground every API claim in the provided excerpts with [n] citations."
            )
        elif label == "video":
            videos = state.get("videos", [])
            if videos:
                vlist = "\n".join(f"- {v.title} — {v.channel} ({v.duration}) {v.url}" for v in videos)
                parts.append(f"Tutorial videos found:\n{vlist}")
                system = (
                    "You are a 2D game-development copilot. Recommend the most relevant of the "
                    "found videos (with their URLs) and say in one line what each covers."
                )
            else:
                parts.append("The video search returned nothing.")
                system = "You are a 2D game-development copilot. Apologize that no videos were found and answer briefly yourself."
        else:
            system = "You are a friendly, concise 2D game-development copilot."

        parts.append(f"User question: {state['question']}")
        return system, "\n\n".join(parts)

    # ---- public API ------------------------------------------------------

    def ask(self, question: str) -> AgentState:
        return self._graph.invoke({"question": question, "path": []})
