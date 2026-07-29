"""YouTube search tool.

GLOSSARY (Tool / tool call): a function registered with the agent. The LLM
cannot execute anything itself — the graph decides (via the router) that a
video is needed, OUR CODE runs the real search, and the results are fed back
into the model's context so the final answer can cite them. In freeform
agents the model emits JSON like {"tool": "youtube_search", "query": ...};
in this LangGraph the tool call is an explicit node, which is easier to see,
test, and debug.

Key-less by design: we fetch YouTube's public results page and parse the
`ytInitialData` JSON blob — no API key, matching the portfolio convention.
Fragile by nature (YouTube can change markup), so it's behind a Protocol with
a Fake for tests and a graceful empty-result failure mode.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True)
class Video:
    title: str
    url: str
    channel: str
    duration: str


class VideoSearch(Protocol):
    def search(self, query: str, k: int = 3) -> list[Video]: ...


class YouTubeSearch:
    def search(self, query: str, k: int = 3) -> list[Video]:
        try:
            r = httpx.get(
                "https://www.youtube.com/results",
                params={"search_query": query},
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"},
                timeout=15.0,
                follow_redirects=True,
            )
            r.raise_for_status()
            m = re.search(r"var ytInitialData = (\{.*?\});</script>", r.text, re.S)
            if not m:
                return []
            data = json.loads(m.group(1))
            return self._extract(data)[:k]
        except Exception:
            return []  # a broken tool degrades the answer, never crashes the graph

    @staticmethod
    def _extract(data: dict) -> list[Video]:
        videos: list[Video] = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                vr = node.get("videoRenderer")
                if isinstance(vr, dict):
                    try:
                        videos.append(
                            Video(
                                title=vr["title"]["runs"][0]["text"],
                                url=f"https://www.youtube.com/watch?v={vr['videoId']}",
                                channel=vr.get("ownerText", {}).get("runs", [{}])[0].get("text", "?"),
                                duration=vr.get("lengthText", {}).get("simpleText", "live/short"),
                            )
                        )
                    except (KeyError, IndexError):
                        pass
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(data)
        return videos


class FakeVideoSearch:
    """Deterministic tool double for tests."""

    def __init__(self, videos: list[Video] | None = None):
        self.videos = videos if videos is not None else [
            Video("Phaser 3 Tilemap Tutorial", "https://www.youtube.com/watch?v=fake1", "GameDevAcademy", "18:22"),
            Video("Arcade Physics Deep Dive", "https://www.youtube.com/watch?v=fake2", "Ourcade", "24:01"),
        ]
        self.queries: list[str] = []

    def search(self, query: str, k: int = 3) -> list[Video]:
        self.queries.append(query)
        return self.videos[:k]
