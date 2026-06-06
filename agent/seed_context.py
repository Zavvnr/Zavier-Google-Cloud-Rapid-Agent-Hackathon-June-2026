"""
Seed `mlangcast.context` with embedded football-context docs, then run a
sample vector search to prove the index works end to end.

This script also doubles as your connection check — if MONGODB_URI is wrong,
the insert will fail loudly.

Prereqs:
    pip install google-genai "pymongo[srv]"
    .env (or exported env vars) with:
        GEMINI_API_KEY=...
        MONGODB_URI=mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/mlangcast?retryWrites=true&w=majority

Run:
    python seed_context.py
"""

import os
import time

from google import genai
from google.genai import types
from pymongo import MongoClient

# --- Config (must line up with what you created in Atlas) ---
EMBED_MODEL = "gemini-embedding-001"
DIMS = 768                 # MUST equal numDimensions in the vector_index
DB_NAME = "mlangcast"
COLLECTION = "context"
INDEX_NAME = "vector_index"

genai_client = genai.Client()                      # reads GEMINI_API_KEY
mongo = MongoClient(os.environ["MONGODB_URI"])
col = mongo[DB_NAME][COLLECTION]


def embed(text: str, task_type: str) -> list[float]:
    """Return a 768-dim embedding.

    task_type matters for retrieval quality: use RETRIEVAL_DOCUMENT for stored
    text and RETRIEVAL_QUERY for the search query. No manual normalization is
    needed because the index uses cosine similarity (magnitude-independent).
    """
    resp = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=DIMS,
            task_type=task_type,
        ),
    )
    return resp.embeddings[0].values


# --- Toy context. Replace / extend with your real player, team, and glossary
#     data (e.g. derived from StatsBomb lineups + a terminology list). ---
CONTEXT_DOCS = [
    {"kind": "player", "name": "Lionel Messi",
     "text": "Lionel Messi, Argentine forward and captain of Argentina, won the "
             "2022 World Cup, celebrated for his dribbling and playmaking."},
    {"kind": "player", "name": "Kylian Mbappe",
     "text": "Kylian Mbappe, French forward known for explosive pace, scored a "
             "hat-trick in the 2022 World Cup final."},
    {"kind": "team", "name": "Argentina",
     "text": "Argentina national football team, three-time World Cup winners "
             "(1978, 1986, 2022), nicknamed La Albiceleste."},
    {"kind": "term", "name": "nutmeg",
     "text": "A nutmeg is when a player plays the ball through an opponent's legs "
             "and collects it on the other side."},
    {"kind": "term", "name": "hat-trick",
     "text": "A hat-trick is when a single player scores three goals in one match."},
]


def seed() -> None:
    col.delete_many({})                            # clean slate so re-runs are safe
    docs = [
        {**d, "embedding": embed(d["text"], "RETRIEVAL_DOCUMENT")}
        for d in CONTEXT_DOCS
    ]
    col.insert_many(docs)
    print(f"Inserted {len(docs)} context docs into {DB_NAME}.{COLLECTION}")


def search(query: str, k: int = 3) -> None:
    qvec = embed(query, "RETRIEVAL_QUERY")
    results = list(col.aggregate([
        {"$vectorSearch": {
            "index": INDEX_NAME,
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": 50,
            "limit": k,
        }},
        {"$project": {
            "_id": 0, "name": 1, "kind": 1,
            "score": {"$meta": "vectorSearchScore"},
        }},
    ]))

    print(f"\nTop {k} for: {query!r}")
    if not results:
        print("  (nothing yet — the index may still be ingesting; wait a few "
              "seconds and call search() again)")
    for r in results:
        print(f"  {r['score']:.3f}  [{r['kind']}] {r['name']}")


if __name__ == "__main__":
    seed()
    time.sleep(5)                                  # give the index a moment to ingest
    search("who scored three goals in the final")  # expect Mbappe / hat-trick on top
