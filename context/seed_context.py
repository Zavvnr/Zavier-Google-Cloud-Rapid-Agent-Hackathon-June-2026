"""
Day 3: seed the MongoDB "mlangcast.context" collection with football-context docs.

Two modes, on purpose:

  * Preview (default, fully offline): turn cached lineups/meta into plain documents
    and print/return them. No network, no env reads on import. This is what the
    placeholder tests exercise — `seed_context_store(docs)` with no client returns a
    "pending" status and inserts nothing.

  * Seed (--seed): connect to Atlas via MONGODB_URI, embed each doc with
    google-genai (gemini-embedding-001, matching agent/seed_context.py and the
    vector_index), and insert. This is the real Day 3 write path.

Document schema (kept identical to agent/seed_context.py so one collection serves
both the seed/search demo and the runtime MongoMCPContextClient):
    {"kind": "player|team|term", "name": str, "text": str, ["team": str],
     ["embedding": [float, ...]]}

CLI:
    # offline preview from cached match files (run data/loader.py first):
    python -m context.seed_context --match-id 3869685
    python -m context.seed_context --lineups data/cache/3869685/lineups.json \
                                   --meta    data/cache/3869685/meta.json

    # real seed into Atlas (needs MONGODB_URI + GOOGLE_API_KEY):
    python -m context.seed_context --match-id 3869685 --seed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent

# Embedding config — MUST line up with the Atlas vector_index and the seed demo.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
DB_NAME = "mlangcast"
COLLECTION = "context"


DEFAULT_GLOSSARY = [
    {
        "kind": "term",
        "name": "hat-trick",
        "text": "A hat-trick is when one player scores three goals in a match.",
    },
    {
        "kind": "term",
        "name": "nutmeg",
        "text": "A nutmeg is when a player plays the ball through an opponent's legs.",
    },
]


def player_docs_from_lineups(lineups: Iterable[dict]) -> list[dict]:
    """Create simple player context documents from StatsBomb lineup JSON."""
    docs = []
    for team in lineups:
        team_name = team.get("team_name") or team.get("team", {}).get("name", "")
        for player in team.get("lineup", []):
            player_name = player.get("player_name") or player.get("name", "")
            if not player_name:
                continue
            docs.append({
                "kind": "player",
                "name": player_name,
                "team": team_name,
                "text": f"{player_name} is listed in the {team_name} match lineup.",
            })
    return docs


def team_docs_from_meta(meta: Optional[dict]) -> list[dict]:
    """Create team context documents from cached StatsBomb match metadata."""
    if not meta:
        return []
    home = (meta.get("home_team") or {}).get("home_team_name", "")
    away = (meta.get("away_team") or {}).get("away_team_name", "")
    docs = []
    for team_name in (home, away):
        if team_name:
            docs.append({
                "kind": "team",
                "name": team_name,
                "text": f"{team_name} are participating in this match.",
            })
    return docs


def build_seed_documents(
    lineups: Optional[Iterable[dict]] = None,
    meta: Optional[dict] = None,
    glossary: Optional[Iterable[dict]] = None,
) -> list[dict]:
    """Combine player, team, and glossary records into Day 3 seed documents."""
    docs = []
    docs.extend(player_docs_from_lineups(lineups or []))
    docs.extend(team_docs_from_meta(meta))
    docs.extend(dict(item) for item in (glossary or DEFAULT_GLOSSARY))
    return docs


def seed_context_store(
    documents: Iterable[dict],
    client=None,
    embedder: Optional[Callable[[str], List[float]]] = None,
    replace: bool = True,
) -> dict:
    """
    Insert seed documents into a collection-like `client`, or preview if none.

    * client is None  -> nothing is written; returns a "pending" preview status.
                         (Day 1/2 + the placeholder tests rely on this.)
    * client provided -> real insert. `client` is anything exposing insert_many()
                         (a pymongo collection in production, a fake in tests).
                         If `embedder` is given, an "embedding" vector is attached
                         to each document before insert. If `replace`, the
                         collection is cleared first so re-runs stay idempotent.
    """
    docs = list(documents)
    if client is None:
        return {"ready": False, "inserted": 0, "pending": len(docs)}

    if embedder is not None:
        docs = [{**doc, "embedding": list(embedder(doc["text"]))} for doc in docs]

    if replace:
        try:
            client.delete_many({})
        except Exception:
            pass  # fresh collections have nothing to clear

    result = client.insert_many(docs)
    inserted = len(getattr(result, "inserted_ids", docs))
    return {"ready": True, "inserted": inserted, "pending": 0}


# --------------------------------------------------------------------------- #
# Real seed helpers (only used by the --seed CLI path)                        #
# --------------------------------------------------------------------------- #
def connect_collection(uri: Optional[str] = None):
    """Open the Atlas mlangcast.context collection (raises loudly on bad config)."""
    from pymongo import MongoClient  # imported lazily so preview mode needs no pymongo

    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI is not set — cannot seed. See README/.env.example.")
    # Use certifi's CA bundle when available. On Windows / brand-new Python builds
    # the system CA store often isn't wired into OpenSSL, which is the usual cause
    # of Atlas "SSL handshake failed (TLSV1_ALERT_INTERNAL_ERROR)".
    kwargs = {}
    try:
        import certifi
        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        pass
    return MongoClient(uri, **kwargs)[DB_NAME][COLLECTION]


def create_vector_index(collection, name: str = "vector_index",
                        dims: int = EMBED_DIMS, similarity: str = "cosine") -> str:
    """
    Create the Atlas Vector Search index that the $vectorSearch query needs.

    Idempotent: if an index of this name already exists, it's left untouched.
    The index builds in the background — it isn't queryable for ~1-2 minutes
    (check with --list-indexes). Requires a recent pymongo (>= 4.7).
    """
    from pymongo.operations import SearchIndexModel

    try:
        existing = {ix.get("name") for ix in collection.list_search_indexes()}
    except Exception:
        existing = set()
    if name in existing:
        return f"Vector index '{name}' already exists — nothing to do."

    model = SearchIndexModel(
        definition={
            "fields": [{
                "type": "vector",
                "path": "embedding",
                "numDimensions": dims,
                "similarity": similarity,
            }]
        },
        name=name,
        type="vectorSearch",
    )
    collection.create_search_index(model=model)
    return (f"Created vector index '{name}' (path=embedding, dims={dims}, "
            f"similarity={similarity}). It builds in the background — give it "
            f"~1-2 minutes, then check status with --list-indexes.")


def list_vector_indexes(collection) -> str:
    """List Atlas search/vector indexes and whether each is queryable yet."""
    rows = []
    for ix in collection.list_search_indexes():
        rows.append(f"  {ix.get('name')}: status={ix.get('status', '?')}, "
                    f"queryable={ix.get('queryable', '?')}")
    return "\n".join(rows) if rows else "  (no search indexes yet)"


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed document texts with gemini-embedding-001 in as FEW API calls as possible.

    Batching matters on the free tier: embedding 54 docs one-per-request trips the
    ~5 requests/minute limit instantly, whereas one batched request stays well
    under it. Inputs are chunked to respect per-request batch limits.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY
    config = types.EmbedContentConfig(
        output_dimensionality=EMBED_DIMS,
        task_type="RETRIEVAL_DOCUMENT",
    )
    vectors: List[List[float]] = []
    for start in range(0, len(texts), 100):  # chunk to stay within request limits
        chunk = texts[start:start + 100]
        resp = client.models.embed_content(model=EMBED_MODEL, contents=chunk, config=config)
        vectors.extend(list(e.values) for e in resp.embeddings)
    return vectors


def _embedding_error_hint(exc: Exception) -> str:
    """Short, actionable message for Gemini embedding failures (esp. quota)."""
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return ("Gemini embedding quota hit (HTTP 429) — the free tier is very low.\n"
                "  - Seed text-only for now (no vectors, no Gemini):\n"
                "      python -m context.seed_context --match-id 3869685 --seed --no-embed\n"
                "  - Or wait ~1 minute for the quota to reset, or enable billing, then retry.")
    return f"Embedding failed: {type(exc).__name__}: {msg[:300]}"


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _load_json(path: Path):
    """Read a JSON document from disk for the CLI."""
    return json.loads(path.read_text(encoding="utf-8"))


def _cached(match_id: int, name: str) -> Optional[Path]:
    """Return a cached match file path if it exists (events/lineups/meta)."""
    path = REPO / "data" / "cache" / str(match_id) / name
    return path if path.exists() else None


def _atlas_error_hint(exc: Exception) -> str:
    """Turn a raw pymongo connection error into a short, actionable message."""
    msg = str(exc)
    lines = ["Could not connect to MongoDB Atlas:",
             f"  {type(exc).__name__}: {msg[:300]}", ""]
    connection_ish = any(k in msg for k in
                         ("SSL", "TLS", "handshake", "ServerSelection", "timed out", "getaddrinfo"))
    if connection_ish:
        lines += [
            "Most common causes (check in this order):",
            "  1. Your IP isn't allow-listed: Atlas > Network Access > Add IP Address",
            "     (your current IP, or 0.0.0.0/0 for a quick hackathon test).",
            "  2. The cluster is paused (free M0 clusters auto-pause) — resume it.",
            "  3. MONGODB_URI wrong: use the SRV string and URL-encode special",
            "     characters in the password (@ : / become %40 %3A %2F).",
            "  4. TLS broken by a firewall/proxy or a pre-release Python: try",
            "     `pip install -U certifi`, or use Python 3.11/3.12 for the venv.",
        ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Preview seed documents, or seed Atlas when --seed is given."""
    parser = argparse.ArgumentParser(description="Build/seed Day 3 context documents.")
    parser.add_argument("--match-id", type=int, default=None,
                        help="Load lineups/meta from data/cache/<id>/.")
    parser.add_argument("--lineups", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--seed", action="store_true",
                        help="Actually write to MongoDB (needs MONGODB_URI + GOOGLE_API_KEY).")
    parser.add_argument("--no-embed", action="store_true",
                        help="Seed without embeddings (text-search only; skips Gemini).")
    parser.add_argument("--create-index", action="store_true",
                        help="Create the Atlas vector_index on the embedding field, then exit.")
    parser.add_argument("--list-indexes", action="store_true",
                        help="List Atlas search/vector indexes and their status, then exit.")
    args = parser.parse_args(argv)

    # Index management actions connect to Atlas but don't need match files.
    if args.create_index or args.list_indexes:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass
        try:
            collection = connect_collection()
            if args.create_index:
                print(create_vector_index(collection))
            if args.list_indexes:
                print("Search indexes:")
                print(list_vector_indexes(collection))
        except Exception as exc:
            print(_atlas_error_hint(exc), file=sys.stderr)
            return 1
        return 0

    # Resolve input files (explicit flags win; otherwise fall back to the cache).
    lineups_path = args.lineups or (_cached(args.match_id, "lineups.json") if args.match_id else None)
    meta_path = args.meta or (_cached(args.match_id, "meta.json") if args.match_id else None)

    lineups = _load_json(lineups_path) if lineups_path else []
    meta = _load_json(meta_path) if meta_path else None
    docs = build_seed_documents(lineups=lineups, meta=meta)

    if not args.seed:
        # Offline preview — no network, no env reads.
        print(json.dumps(docs, indent=2, ensure_ascii=False))
        print(f"\n# preview only — {len(docs)} docs. Add --seed to write to Atlas.")
        return 0

    # Load .env so MONGODB_URI / GOOGLE_API_KEY are available (matches how the
    # agent/pipeline/web entrypoints read their own config for real runs).
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass

    # Real seed path: connect -> (batch-embed) -> insert. Each step reports its own
    # failure clearly, so a Gemini quota error isn't mislabelled as a Mongo error.
    try:
        collection = connect_collection()
    except Exception as exc:
        print(_atlas_error_hint(exc), file=sys.stderr)
        return 1

    if not args.no_embed:
        try:
            vectors = embed_documents([doc["text"] for doc in docs])
            docs = [{**doc, "embedding": vec} for doc, vec in zip(docs, vectors)]
        except Exception as exc:
            print(_embedding_error_hint(exc), file=sys.stderr)
            return 1

    try:
        status = seed_context_store(docs, client=collection)
    except Exception as exc:
        print(_atlas_error_hint(exc), file=sys.stderr)  # Mongo write/connection
        return 1

    print(f"Seeded {status['inserted']} docs into {DB_NAME}.{COLLECTION} "
          f"({'without' if args.no_embed else 'with'} embeddings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
