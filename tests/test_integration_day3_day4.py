"""
Integration tests for Day 3 (MongoDB context retrieval + seeding) and Day 4
(Google Cloud TTS + pipeline wiring + web app).

These exercise the REAL code paths — vector/text search, embedding-on-seed, the
TTS REST flow, the pipeline, and the Flask routes — without needing a live
MongoDB, a Google API key, or network access. The seams are filled with small
fakes (a fake collection, a fake embedder, a fake HTTP transport), exactly the
way the production objects expose them.

Run:
    python -m unittest tests.test_integration_day3_day4 -v
    # or the whole suite:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.commentary_agent import CommentaryAgent
from agent.mcp_client import MongoMCPContextClient, NoOpContextClient
from context.seed_context import build_seed_documents, seed_context_store
from pipeline.commentary_pipeline import stream_commentary
from tts.speak import GoogleCloudSpeaker, NoOpSpeaker, build_speaker

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #
class _FakeCursor(list):
    """A list that also supports .limit() like a pymongo cursor."""

    def limit(self, n):
        return _FakeCursor(self[:n])


class FakeCollection:
    """Minimal stand-in for a pymongo collection (aggregate/find/insert/delete)."""

    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.inserted = []

    def aggregate(self, pipeline):           # vector search path
        return list(self.docs)

    def find(self, query, projection=None):  # text search path
        return _FakeCursor(self.docs)

    def insert_many(self, docs):
        self.inserted.extend(docs)

        class _Result:
            inserted_ids = list(range(len(docs)))

        return _Result()

    def delete_many(self, query):
        self.inserted.clear()


def _shot_goal(index: int = 1) -> dict:
    """A minimal 'Messi scores' shot event used across the tests."""
    return {
        "index": index, "period": 1, "timestamp": "00:22:10.000",
        "minute": 22, "second": 10,
        "type": {"name": "Shot"}, "team": {"name": "Argentina"},
        "player": {"name": "Lionel Messi"},
        "shot": {"outcome": {"name": "Goal"}, "body_part": {"name": "Left Foot"}},
    }


# --------------------------------------------------------------------------- #
# Day 3 — context retrieval                                                   #
# --------------------------------------------------------------------------- #
class Day3ContextRetrieval(unittest.TestCase):
    def test_vector_search_returns_grouped_context(self):
        """With an embedder + collection, vector results are grouped by kind."""
        col = FakeCollection([
            {"kind": "player", "name": "Lionel Messi", "text": "Argentine forward and captain."},
            {"kind": "team", "name": "Argentina", "text": "La Albiceleste, 2022 champions."},
        ])
        client = MongoMCPContextClient(
            mongodb_uri="mongodb://x",
            collection_handle=col,
            embedder=lambda q: [0.0] * 768,   # pretend embedding -> triggers $vectorSearch
        )
        ctx = client.fetch_event_context(_shot_goal(), {"clock": "22:10", "score": "Argentina 0-0 France"})
        self.assertIn("players", ctx)
        self.assertIn("teams", ctx)
        self.assertTrue(any("Lionel Messi" in line for line in ctx["players"]))

    def test_text_search_fallback_without_embeddings(self):
        """No embedder (no API key) -> keyword search path still returns context."""
        col = FakeCollection([{"kind": "term", "name": "nutmeg", "text": "ball through the legs"}])
        # embedder returns no vector -> retrieval falls back to keyword search,
        # deterministically and without touching the embeddings API.
        client = MongoMCPContextClient(
            mongodb_uri="mongodb://x", collection_handle=col, embedder=lambda q: None
        )
        ctx = client.fetch_event_context(_shot_goal(), {"clock": "22:10"})
        self.assertIn("glossary", ctx)

    def test_unconfigured_clients_return_empty(self):
        """No URI / no-op clients must never break the loop — they return {}."""
        self.assertEqual(MongoMCPContextClient().fetch_event_context(_shot_goal()), {})
        self.assertEqual(NoOpContextClient().fetch_event_context(_shot_goal()), {})

    def test_agent_uses_injected_context_client(self):
        """The agent's Day 3 seam calls the context client and returns its data."""
        class StubContext:
            def fetch_event_context(self, ev, state=None):
                return {"players": ["Lionel Messi: Argentina captain"]}

        agent = CommentaryAgent(language="en", mock=True, context_client=StubContext())
        self.assertEqual(agent.fetch_context(_shot_goal()), {"players": ["Lionel Messi: Argentina captain"]})


# --------------------------------------------------------------------------- #
# Day 3 — seeding                                                             #
# --------------------------------------------------------------------------- #
class Day3Seeding(unittest.TestCase):
    def test_seed_inserts_documents_with_embeddings(self):
        """seed_context_store with a collection + embedder writes embedded docs."""
        docs = build_seed_documents(
            lineups=[{"team_name": "Argentina", "lineup": [{"player_name": "Lionel Messi"}]}],
            meta={"home_team": {"home_team_name": "Argentina"},
                  "away_team": {"away_team_name": "France"}},
        )
        col = FakeCollection()
        status = seed_context_store(docs, client=col, embedder=lambda t: [0.1] * 8)

        self.assertTrue(status["ready"])
        self.assertEqual(status["inserted"], len(docs))
        self.assertTrue(all("embedding" in d for d in col.inserted))

    def test_seed_preview_without_client_writes_nothing(self):
        """No client -> preview status, nothing inserted (offline-safe contract)."""
        docs = build_seed_documents(meta={"home_team": {"home_team_name": "Argentina"},
                                          "away_team": {"away_team_name": "France"}})
        status = seed_context_store(docs)
        self.assertFalse(status["ready"])
        self.assertEqual(status["pending"], len(docs))


# --------------------------------------------------------------------------- #
# Day 4 — TTS                                                                 #
# --------------------------------------------------------------------------- #
class Day4Tts(unittest.TestCase):
    def test_google_speaker_writes_audio_via_injected_transport(self):
        """A fake REST transport returns base64 audio; the speaker decodes + saves it."""
        fake_audio = b"ID3-fake-mp3-bytes"

        def transport(url, payload, key):
            # Assert the request is well-formed, then return a Cloud-TTS-shaped reply.
            assert payload["input"]["text"]
            assert payload["voice"]["languageCode"] == "es-ES"
            return {"audioContent": base64.b64encode(fake_audio).decode()}

        with tempfile.TemporaryDirectory() as tmp:
            speaker = GoogleCloudSpeaker(api_key="test-key", transport=transport)
            result = speaker.synthesize("¡Gol de Argentina!", language="es-ES", output_dir=Path(tmp))
            self.assertTrue(result.has_audio())
            self.assertEqual(result.audio_bytes, fake_audio)
            self.assertTrue(result.audio_path.exists())
            self.assertEqual(result.mime_type, "audio/mpeg")

    def test_google_speaker_failsafe_without_key(self):
        """No key in the environment + no transport -> text-only, no network call."""
        # Clear env so api_key=None can't fall back to a real GOOGLE_API_KEY in the
        # dev's shell. This exercises the true "no credentials" path deterministically.
        with mock.patch.dict(os.environ, {}, clear=True):
            result = GoogleCloudSpeaker().synthesize("hello", language="en-US")
        self.assertFalse(result.has_audio())
        self.assertIn("GOOGLE_API_KEY", result.skipped_reason)

    def test_build_speaker_provider_selection(self):
        """Provider routing: google when asked, noop otherwise."""
        self.assertIsInstance(build_speaker(enabled=True, provider="google"), GoogleCloudSpeaker)
        self.assertIsInstance(build_speaker(enabled=True), NoOpSpeaker)
        self.assertIsInstance(build_speaker(enabled=False, provider="google"), NoOpSpeaker)


# --------------------------------------------------------------------------- #
# Day 4 — pipeline end to end                                                 #
# --------------------------------------------------------------------------- #
class Day4Pipeline(unittest.TestCase):
    def test_pipeline_attaches_audio_when_speaker_injected(self):
        """stream_commentary should pass agent text to the speaker and carry audio out."""
        def transport(url, payload, key):
            return {"audioContent": base64.b64encode(b"MP3").decode()}

        speaker = GoogleCloudSpeaker(api_key="test-key", transport=transport)
        outputs = list(stream_commentary([_shot_goal()], language="es", speed=0, mock=True, speaker=speaker))

        self.assertEqual(len(outputs), 1)
        self.assertIn("GOAL", outputs[0].text)
        self.assertTrue(outputs[0].speech.has_audio())
        self.assertEqual(outputs[0].speech.language, "es-ES")   # "es" normalized
        self.assertEqual(outputs[0].as_dict()["event_type"], "Shot")


class RealMatchSmoke(unittest.TestCase):
    """End-to-end over the real cached WC2022 final (skipped if not downloaded)."""

    def test_wc2022_final_mock_pipeline(self):
        events_path = REPO / "data" / "cache" / "3869685" / "events.json"
        if not events_path.exists():
            self.skipTest("WC2022 final not cached (run data/loader.py --match-id 3869685)")
        events = json.loads(events_path.read_text(encoding="utf-8"))

        outputs = list(stream_commentary(events, language="es", speed=0, mock=True))

        # The agent's pacing should collapse thousands of events to a readable feed,
        # while still catching the goals (the final had six in regulation).
        self.assertGreater(len(outputs), 20)
        self.assertLess(len(outputs), len(events))
        goals = [o for o in outputs if "GOAL" in o.text]
        self.assertGreaterEqual(len(goals), 6)
        self.assertTrue(all(o.speech.language == "es-ES" for o in outputs))


# --------------------------------------------------------------------------- #
# Day 4 — web app (Flask routes + SSE)                                        #
# --------------------------------------------------------------------------- #
class WebApp(unittest.TestCase):
    """Exercise the Flask routes with the test client (skipped if Flask absent)."""

    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed (pip install flask)")
        from web.app import create_app
        self.client = create_app().test_client()

    def test_index_serves_ui(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"MlangCast", resp.data)

    def test_languages_and_matches_endpoints(self):
        langs = self.client.get("/api/languages").get_json()
        self.assertTrue(any(item["code"] == "es-ES" for item in langs))
        matches = self.client.get("/api/matches").get_json()
        ids = {m["id"] for m in matches}
        self.assertIn("sample", ids)  # bundled sample is always offered

    def test_stream_sample_in_mock_mode_emits_goal_and_done(self):
        resp = self.client.get("/api/stream?match=sample&language=es-ES&mock=1&speed=0&tts=0")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("GOAL", body)            # the sample build-up ends in a goal
        self.assertIn("event: done", body)     # stream terminates cleanly


if __name__ == "__main__":
    unittest.main()
