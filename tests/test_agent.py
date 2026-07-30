"""Hermetic agent tests — the whole graph, zero models, zero network.

FakeLLM scripts the model, HashEmbedder + ephemeral Chroma make retrieval
real-but-deterministic, FakeVideoSearch doubles the tool. If these pass, the
architecture works; Ollama only changes the words, not the wiring.
"""
from __future__ import annotations

import pytest

from copilot.domain import GAMEDEV, DomainProfile, load_profile
from copilot.graph import Copilot
from copilot.llm.fakes import FakeLLM, HashEmbedder
from copilot.memory.memory import ConversationBuffer, ProjectMemory
from copilot.nodes.router import heuristic_route, route
from copilot.rag.chunking import chunk_text
from copilot.rag.store import Retriever
from copilot.tools.youtube import FakeVideoSearch, Video, YouTubeSearch


# ---------- router ----------------------------------------------------------

def test_router_parses_clean_and_messy_llm_output():
    assert route(FakeLLM(["docs"]), "how do tilemap collisions work?") == "docs"
    assert route(FakeLLM(["  Video  "]), "any tutorial?") == "video"
    assert route(FakeLLM(["I think this is a docs question"]), "what is a sprite?") == "docs"


def test_router_falls_back_to_heuristic_on_garbage():
    assert route(FakeLLM(["banana"]), "show me a tutorial video for tilemaps") == "video"
    assert route(FakeLLM(["???"]), "how does arcade physics gravity work") == "docs"
    assert heuristic_route("hola!") == "chat"


def test_router_heuristic_uses_domain_keywords():
    """Subject words are DomainProfile config, not hardcoded vocabulary."""
    assert heuristic_route("tilemap collisions again", GAMEDEV.docs_keywords) == "docs"
    astro = ("telescope", "nebula", "aperture")
    assert heuristic_route("best aperture for nebula shots", astro) == "docs"
    assert heuristic_route("best aperture for nebula shots") == "chat"  # no profile → no match


# ---------- chunking --------------------------------------------------------

def test_chunking_splits_and_overlaps():
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 120) for i in range(8))
    chunks = chunk_text(text, source="test://doc", max_words=300, overlap_words=40)
    assert len(chunks) > 2
    assert all(c.source == "test://doc" for c in chunks)
    # overlap: some tail words of chunk n appear in chunk n+1
    assert set(chunks[0].text.split()[-10:]) & set(chunks[1].text.split())


# ---------- retrieval (real Chroma, deterministic embeddings) --------------

@pytest.fixture()
def retriever() -> Retriever:
    r = Retriever(HashEmbedder())
    docs = {
        "phaser://tilemaps": "Tilemap collision in Phaser uses setCollision and collider objects between layers and sprites.",
        "phaser://physics": "Arcade physics gives bodies velocity gravity and acceleration for fast 2D games.",
        "phaser://animations": "Sprite animations are created with anims createFrames and played on a sprite.",
    }
    for src, text in docs.items():
        r.add(chunk_text(text, source=src))
    return r


def test_retriever_ranks_by_meaning(retriever: Retriever):
    hits = retriever.search("how do I set collision on a tilemap layer?", k=2)
    assert hits and hits[0].source == "phaser://tilemaps"


def test_retriever_empty_index_returns_no_hits():
    assert Retriever(HashEmbedder()).search("anything") == []


# ---------- full graph paths ------------------------------------------------

def make_bot(llm: FakeLLM, retriever: Retriever, videos: FakeVideoSearch | None = None,
             project: ProjectMemory | None = None) -> Copilot:
    return Copilot(
        llm=llm,
        embedder=HashEmbedder(),
        retriever=retriever,
        video_search=videos or FakeVideoSearch(),
        buffer=ConversationBuffer(max_turns=4),
        project=project,
    )


def test_docs_question_takes_retrieve_path_and_grounds_answer(retriever: Retriever):
    llm = FakeLLM(["docs", "Use layer.setCollision as shown in [1]."])
    bot = make_bot(llm, retriever)
    state = bot.ask("how do I do tilemap collisions?")
    assert state["path"] == ["route", "retrieve", "generate", "update_memory"]
    assert state["hits"][0].source == "phaser://tilemaps"
    # the generate prompt actually contained the retrieved chunk (grounding)
    assert "setCollision" in llm.calls[-1]["prompt"]
    assert "[1]" in state["answer"]


def test_video_question_takes_tool_path(retriever: Retriever):
    videos = FakeVideoSearch()
    llm = FakeLLM(["video", "Watch 'Phaser 3 Tilemap Tutorial' by GameDevAcademy."])
    bot = make_bot(llm, retriever, videos=videos)
    state = bot.ask("got a tutorial video for tilemaps?")
    assert state["path"] == ["route", "search_video", "generate", "update_memory"]
    assert videos.queries == ["got a tutorial video for tilemaps?"]
    assert state["videos"][0].url.startswith("https://www.youtube.com/")


def test_chat_question_skips_both_tools(retriever: Retriever):
    llm = FakeLLM(["chat", "Hello! Ready to build."])
    state = make_bot(llm, retriever).ask("hey there")
    assert state["path"] == ["route", "generate", "update_memory"]
    assert "hits" not in state or not state["hits"]


def test_conversation_buffer_carries_context(retriever: Retriever):
    llm = FakeLLM(["docs", "Set gravity on the body [1].", "docs", "Lower gravity makes the jump floatier [1]."])
    bot = make_bot(llm, retriever)
    bot.ask("how does arcade physics gravity work?")
    bot.ask("and how do I make that jump feel better?")
    # short-term memory: the second generate call saw the first exchange
    assert "gravity" in llm.calls[-1]["prompt"]
    assert "Recent conversation:" in llm.calls[-1]["prompt"]


def test_project_memory_persists_and_is_injected(tmp_path, retriever: Retriever):
    mem = ProjectMemory(tmp_path / "mem.json")
    mem.remember("project uses Phaser 3 + TypeScript, arcade physics")
    llm = FakeLLM(["chat", "Noted!"])
    make_bot(llm, retriever, project=mem).ask("hi")
    assert "arcade physics" in llm.calls[-1]["prompt"]
    # survives a "new session"
    assert ProjectMemory(tmp_path / "mem.json").facts() == ["project uses Phaser 3 + TypeScript, arcade physics"]


def test_docs_path_with_empty_index_admits_it(tmp_path):
    llm = FakeLLM(["docs", "The docs index has no coverage of that; generally speaking..."])
    bot = make_bot(llm, Retriever(HashEmbedder()))
    state = bot.ask("how do tweens work?")
    assert state["path"] == ["route", "retrieve", "generate", "update_memory"]
    assert "No documentation matched" in llm.calls[-1]["prompt"]


# ---------- domain profiles: the same graph researches ANY subject ----------

def test_custom_domain_profile_reskins_every_prompt(retriever: Retriever):
    astronomy = DomainProfile(
        name="astronomy",
        banner="🔭 research-copilot · astronomy",
        persona="a concise astronomy research copilot",
        docs_keywords=("telescope", "nebula"),
    )
    llm = FakeLLM(["docs", "Dust and ionized gases [1]."])
    bot = Copilot(
        llm=llm, embedder=HashEmbedder(), retriever=retriever,
        video_search=FakeVideoSearch(), domain=astronomy,
    )
    state = bot.ask("what is a nebula made of?")
    assert state["path"] == ["route", "retrieve", "generate", "update_memory"]
    assert "astronomy research copilot" in llm.calls[-1]["system"]
    assert "game" not in llm.calls[-1]["system"].lower()


def test_load_profile_reads_json_and_defaults_to_gamedev(tmp_path):
    assert load_profile(None) == GAMEDEV
    p = tmp_path / "fastapi.json"
    p.write_text(
        '{"name": "FastAPI", "banner": "b", "persona": "a concise FastAPI research copilot",'
        ' "docs_keywords": ["pydantic"], "urls_file": "corpus/fastapi.txt"}',
        encoding="utf-8",
    )
    prof = load_profile(str(p))
    assert prof.name == "FastAPI"
    assert prof.docs_keywords == ("pydantic",)
    assert prof.urls_file == "corpus/fastapi.txt"


# ---------- youtube parser (offline, fixture data) --------------------------

def test_youtube_extract_parses_renderer_shape():
    data = {"a": [{"videoRenderer": {
        "videoId": "abc123",
        "title": {"runs": [{"text": "Phaser Tilemap Guide"}]},
        "ownerText": {"runs": [{"text": "Ourcade"}]},
        "lengthText": {"simpleText": "12:34"},
    }}]}
    vids = YouTubeSearch._extract(data)
    assert vids == [Video("Phaser Tilemap Guide", "https://www.youtube.com/watch?v=abc123", "Ourcade", "12:34")]


def test_ask_streams_node_completions_in_order(retriever: Retriever):
    llm = FakeLLM(["docs", "Grounded answer [1]."])
    bot = make_bot(llm, retriever)
    seen: list[str] = []
    state = bot.ask("how do tilemap collisions work?", on_node=seen.append)
    assert seen == ["route", "retrieve", "generate", "update_memory"]
    assert state["answer"] == "Grounded answer [1]."
    assert state["path"] == seen  # callback saw exactly the real path
