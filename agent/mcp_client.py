"""
MongoDB context retrieval for the commentary agent (the partner "MCP" seam).

The agent calls a context client as a tool to pull player/team/standings/glossary
facts for the current event. Two implementations:

  * NoOpContextClient    — returns {} (Day 1/2 stays fully offline).
  * MongoMCPContextClient — real Atlas Vector Search retrieval, mirroring the
                            schema seeded by agent/seed_context.py
                            (db "mlangcast", collection "context", index
                            "vector_index", path "embedding", gemini-embedding-001).

IMPORTANT — fail-safe by design: every retrieval path is wrapped so that a missing
dependency (pymongo / google-genai), a bad/unreachable MONGODB_URI, a missing API
key, or an empty result ALWAYS degrades to {} instead of breaking the live loop.
That property is what keeps the offline placeholder tests green even now that the
real logic is wired.

How it's used (see agent/commentary_agent.py -> fetch_context):
    client = build_context_client(enabled=True)   # MongoMCPContextClient()
    ctx = client.fetch_event_context(event, state)  # {} or grouped context

The cloud Agent Builder agent reaches the same data through MongoDB's MCP server;
this module is the equivalent retrieval seam for the local Python pipeline. Both
read the same collection, so seeding once (context/seed_context.py or
agent/seed_context.py) serves both.

Env vars used (names only; this module never reads .env files directly):
    MONGODB_URI   — Atlas connection string
    GOOGLE_API_KEY (or GEMINI_API_KEY) — used by google-genai for query embeddings
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


def _name(value: Optional[dict], key: str = "name") -> str:
    """Safely extract a named field from a StatsBomb-style nested dictionary."""
    if not value:
        return ""
    return str(value.get(key) or "").strip()


def build_context_query(event: dict, state: Optional[dict] = None) -> str:
    """Build the search text sent to MongoDB vector/MCP retrieval for an event."""
    state = state or {}
    etype = _name(event.get("type"))
    team = _name(event.get("team"))
    player = _name(event.get("player"))
    score = str(state.get("score") or "").strip()
    clock = str(state.get("clock") or "").strip()
    parts = [part for part in (clock, score, etype, team, player) if part]

    if etype == "Shot":
        shot = event.get("shot") or {}
        parts.append(_name(shot.get("outcome")))
        parts.append(_name(shot.get("body_part")))
    elif etype == "Pass":
        pass_data = event.get("pass") or {}
        parts.append(_name(pass_data.get("recipient")))
        parts.append(_name(pass_data.get("technique")))
    elif etype == "Substitution":
        substitution = event.get("substitution") or {}
        parts.append(_name(substitution.get("replacement")))

    return " ".join(part for part in parts if part)


# Map a seed document's "kind" to a friendly group key for the prompt context.
_KIND_TO_GROUP = {"player": "players", "team": "teams", "term": "glossary"}


def _format_context(docs: Optional[List[dict]]) -> dict:
    """Turn retrieved docs into a compact {group: [lines]} dict (or {} if none)."""
    groups: dict = {}
    for doc in docs or []:
        group = _KIND_TO_GROUP.get(doc.get("kind", ""), "context")
        name = (doc.get("name") or "").strip()
        text = (doc.get("text") or "").strip()
        line = f"{name}: {text}".strip(": ").strip() if (name or text) else ""
        if line:
            groups.setdefault(group, []).append(line)
    return groups


@dataclass
class NoOpContextClient:
    """Context client used when Day 3 retrieval is disabled; returns nothing."""

    reason: str = "MongoDB context retrieval is disabled for this run."

    def fetch_event_context(self, event: dict, state: Optional[dict] = None) -> dict:
        """Return empty context so the Day 1/2 pipeline stays fully offline-safe."""
        return {}


@dataclass
class MongoMCPContextClient:
    """
    Real Atlas Vector Search retrieval for the commentary agent (Day 3).

    Defaults match agent/seed_context.py so a single seeding pass serves both the
    seed/search demo and this runtime client. For tests, inject `collection_handle`
    (anything exposing .aggregate/.find) and/or `embedder` (a callable returning a
    vector) to exercise the real query logic without a live MongoDB or API key.
    """

    mongodb_uri: Optional[str] = None
    database: str = "mlangcast"
    collection: str = "context"          # collection NAME (not a handle)
    index_name: str = "vector_index"
    embed_model: str = "gemini-embedding-001"
    dims: int = 768
    limit: int = 3
    num_candidates: int = 50
    server_timeout_ms: int = 800         # fail fast on a bad/unreachable URI
    # Test/seam injection points (left None in production):
    collection_handle: object = None
    embedder: Optional[Callable[[str], List[float]]] = None

    def __post_init__(self) -> None:
        """Fill the URI from the environment without reading env files directly."""
        if self.mongodb_uri is None:
            self.mongodb_uri = os.getenv("MONGODB_URI")
        self._live_collection = None  # cached pymongo collection after first connect

    def is_configured(self) -> bool:
        """Report whether there is enough configuration to attempt a connection."""
        return bool(self.mongodb_uri) or self.collection_handle is not None

    # -- connection / embedding (each fail-safe) ---------------------------- #
    def _get_collection(self):
        """Return a collection handle (injected, cached, or freshly connected)."""
        if self.collection_handle is not None:
            return self.collection_handle
        if self._live_collection is not None:
            return self._live_collection
        if not self.mongodb_uri:
            return None
        try:
            from pymongo import MongoClient  # optional dependency
            kwargs = {"serverSelectionTimeoutMS": self.server_timeout_ms}
            try:  # certifi CA bundle fixes most Windows/new-Python Atlas TLS failures
                import certifi
                kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass
            client = MongoClient(self.mongodb_uri, **kwargs)
            client.admin.command("ping")  # raises quickly if unreachable
            self._live_collection = client[self.database][self.collection]
            return self._live_collection
        except Exception:
            return None

    def _embed(self, query: str) -> Optional[List[float]]:
        """Embed the query for vector search; None if embeddings are unavailable."""
        if self.embedder is not None:
            try:
                return list(self.embedder(query))
            except Exception:
                return None
        try:
            from google import genai
            from google.genai import types
            client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY
            resp = client.models.embed_content(
                model=self.embed_model,
                contents=query,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dims,
                    task_type="RETRIEVAL_QUERY",
                ),
            )
            return list(resp.embeddings[0].values)
        except Exception:
            return None

    # -- search strategies -------------------------------------------------- #
    def _vector_search(self, col, query_vector: List[float]) -> List[dict]:
        """Atlas $vectorSearch over the seeded `embedding` field."""
        pipeline = [
            {"$vectorSearch": {
                "index": self.index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": self.num_candidates,
                "limit": self.limit,
            }},
            {"$project": {
                "_id": 0, "name": 1, "kind": 1, "text": 1, "team": 1,
                "score": {"$meta": "vectorSearchScore"},
            }},
        ]
        return list(col.aggregate(pipeline))

    def _text_search(self, col, query: str) -> List[dict]:
        """Keyword fallback when embeddings/vector index are unavailable."""
        tokens = [t for t in re.split(r"\W+", query) if len(t) > 2]
        if not tokens:
            return []
        pattern = "|".join(re.escape(t) for t in tokens)
        cursor = col.find(
            {"$or": [
                {"name": {"$regex": pattern, "$options": "i"}},
                {"text": {"$regex": pattern, "$options": "i"}},
            ]},
            {"_id": 0, "name": 1, "kind": 1, "text": 1, "team": 1},
        )
        try:  # real pymongo cursors support .limit(); fakes may not
            cursor = cursor.limit(self.limit)
        except Exception:
            pass
        return list(cursor)

    # -- public API --------------------------------------------------------- #
    def fetch_event_context(self, event: dict, state: Optional[dict] = None) -> dict:
        """
        Retrieve context for one event. Tries vector search, falls back to keyword
        search, and ALWAYS returns {} on any failure (missing deps, bad URI, no
        results) so the commentary loop never breaks.
        """
        query = build_context_query(event, state)
        if not query:
            return {}
        try:
            col = self._get_collection()
            if col is None:
                return {}
            docs: List[dict] = []
            query_vector = self._embed(query)
            if query_vector is not None:
                docs = self._vector_search(col, query_vector)
            if not docs:
                docs = self._text_search(col, query)
            return _format_context(docs)
        except Exception:
            # Retrieval is best-effort; never propagate to the live loop.
            return {}


def build_context_client(enabled: bool = False) -> object:
    """Create the context client the pipeline should use for this run."""
    if enabled:
        return MongoMCPContextClient()
    return NoOpContextClient()
