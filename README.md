# 🎮 gamedev-copilot

**A 2D game-dev research copilot** — ask it Phaser questions, and a LangGraph
agent decides whether to answer from indexed docs (RAG), find you a tutorial
video, or just chat — with **every LLM call running locally** (Ollama +
Mistral) and every architectural concept mapped to one readable file.

```
you> How do I make my player collide with a tilemap layer?

[path: route → retrieve → generate → update_memory]  [label: docs]
  [1] docs.phaser.io/api-documentation/class/tilemaps-tilemap  (score 0.71)
  [2] docs.phaser.io/phaser/concepts/physics/arcade            (score 0.66)

copilot> Use this.physics.add.collider(player, layer) after
layer.setCollisionByExclusion([-1]) … [1][2]
```

## Learn the vocabulary by reading the code

> Deep dive: **[ARCHITECTURE.md](ARCHITECTURE.md)** — how every term cooperates
> in one query, with sequence + dependency diagrams.

This repo is deliberately organized so each agent concept lives in ONE small
file with a `GLOSSARY` docstring explaining it in plain terms:

| Term | What it means (short) | Read this file |
|---|---|---|
| **Router node** | classification only: `docs` / `video` / `chat`, graph branches on it | [`copilot/nodes/router.py`](copilot/nodes/router.py) |
| **Indexing** | one-time prep: split + store docs before any question | [`scripts/index_docs.py`](scripts/index_docs.py) |
| **Chunks** | ~400-word passages — models retrieve better over small focused pieces | [`copilot/rag/chunking.py`](copilot/rag/chunking.py) |
| **Embeddings** | text → vector of numbers capturing *meaning* | [`copilot/llm/base.py`](copilot/llm/base.py) |
| **Chroma (vector DB)** | stores embeddings, answers "closest chunks to this query?" in ms | [`copilot/rag/store.py`](copilot/rag/store.py) |
| **Retriever** | embed question → ask Chroma → return top chunks | [`copilot/rag/store.py`](copilot/rag/store.py) |
| **Hallucination** | model invents an API; RAG grounds answers in retrieved text + citations | prompt in [`copilot/graph.py`](copilot/graph.py) |
| **Inference / local inference** | running the model; here on YOUR machine via Ollama | [`copilot/llm/ollama_client.py`](copilot/llm/ollama_client.py) |
| **Ollama / Mistral** | local model server / the 7B model it serves | [`copilot/llm/ollama_client.py`](copilot/llm/ollama_client.py) |
| **Tool / tool call** | the LLM can't execute anything — our code runs the real YouTube search and feeds results back | [`copilot/tools/youtube.py`](copilot/tools/youtube.py) |
| **Conversation buffer** | models remember nothing — recent turns are re-sent every call | [`copilot/memory/memory.py`](copilot/memory/memory.py) |
| **Long-term memory / context management** | durable facts ("Phaser 3 + arcade physics") always injected; old chatter dropped | [`copilot/memory/memory.py`](copilot/memory/memory.py) |
| **Graph / nodes / edges** | explicit workflow you can see and debug — vs a freeform agent | [`copilot/graph.py`](copilot/graph.py) |

```
route ──docs──► retrieve ─────┐
  │───video──► search_video ──┼──► generate ──► update_memory ──► END
  └───chat─────────────────────┘
```

Every answer prints `state["path"]` — the nodes that actually ran — so you
watch the graph work instead of trusting a diagram.

## Testable without any model — the core design decision

Every non-deterministic dependency sits behind a tiny interface with a
deterministic fake:

| Real | Fake (tests) |
|---|---|
| `OllamaLLM` (Mistral) | `FakeLLM` — scripted responses, records every prompt |
| `OllamaEmbedder` (nomic-embed-text) | `HashEmbedder` — stable bag-of-words vectors, real cosine ranking |
| `YouTubeSearch` (live scrape) | `FakeVideoSearch` — canned videos, records queries |
| Persistent Chroma | Ephemeral Chroma (unique per-test collections) |

**12 hermetic tests run the entire graph in ~1 second with zero network and
zero models** — they assert the path taken, the grounding (retrieved chunk
text really appears in the generate prompt), memory injection and
persistence, router salvage of messy LLM output, and the YouTube parser
against fixture data. Ollama changes the words, never the wiring.

```bash
pytest            # 12/12, no Ollama needed
```

## Run the real thing

```bash
# 1. install Ollama (https://ollama.com), then:
ollama pull mistral && ollama pull nomic-embed-text

# 2. index the real Phaser docs (10 pages → ~148 chunks, one-time):
python scripts/index_docs.py

# 3. chat:
python -m copilot.chat
#   /remember project uses Phaser 3 + TypeScript, arcade physics
#   /facts   /forget <n>   /quit
```

Verified end-to-end: a tilemap-collision question routed `docs`, retrieved
the Tilemap API class page + arcade-physics concepts, and produced a cited
answer in ~68 s of pure CPU inference.

## Honesty notes

- **7B is 7B**: even grounded, Mistral occasionally paraphrases an API
  imprecisely — the citations exist precisely so you check the source. A
  larger local model (or stricter quote-only prompting) improves this.
- The YouTube tool scrapes the public results page (key-less by design);
  YouTube can change markup, so it degrades to empty results, never crashes.
- The corpus is 10 real `docs.phaser.io` pages, listed in
  [`corpus/urls.txt`](corpus/urls.txt) — extend it and re-run the indexer.

## What I'd change at scale

- Answer streaming (Ollama supports it) for perceived latency.
- A golden-set retrieval eval (query → expected source) run in CI.
- Web UI that renders the graph path visually; recorded-transcript demo mode
  for GitHub Pages (local inference can't run there).
- Summarization node when the buffer overflows, instead of plain drop-off.
