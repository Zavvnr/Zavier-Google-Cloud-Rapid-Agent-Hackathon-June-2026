"""Tests for Day 3/Day 4 placeholder pipeline seams."""

from __future__ import annotations

import unittest

from agent.mcp_client import (
    MongoMCPContextClient,
    NoOpContextClient,
    build_context_query,
)
from context.seed_context import build_seed_documents, seed_context_store
from pipeline.commentary_pipeline import stream_commentary
from tts.speak import NoOpSpeaker


def shot_event(index: int, outcome: str = "Goal") -> dict:
    """Build a minimal shot event used by the placeholder integration tests."""
    return {
        "index": index,
        "period": 1,
        "timestamp": "00:22:08.000",
        "minute": 22,
        "second": 8,
        "type": {"name": "Shot"},
        "team": {"name": "Argentina"},
        "player": {"name": "Lionel Messi"},
        "shot": {"outcome": {"name": outcome}, "body_part": {"name": "Left Foot"}},
    }


class Day3PlaceholderTests(unittest.TestCase):
    """Verify the MongoDB MCP placeholder remains safe and useful offline."""

    def test_build_context_query_extracts_event_and_state_terms(self) -> None:
        """A future vector search query should include event facts and match state."""
        query = build_context_query(
            shot_event(1, "Saved"),
            {"clock": "22:08", "score": "Argentina 0-0 France"},
        )

        self.assertIn("22:08", query)
        self.assertIn("Shot", query)
        self.assertIn("Lionel Messi", query)
        self.assertIn("Saved", query)

    def test_context_clients_return_empty_context_until_day3_is_wired(self) -> None:
        """No-op and Mongo placeholders should not leak placeholder text to Gemini."""
        event = shot_event(1)

        self.assertEqual(NoOpContextClient().fetch_event_context(event), {})
        self.assertEqual(MongoMCPContextClient(mongodb_uri="mongodb://example").fetch_event_context(event), {})
        self.assertTrue(MongoMCPContextClient(mongodb_uri="mongodb://example").is_configured())

    def test_seed_context_documents_preview_lineups_meta_and_glossary(self) -> None:
        """Seed previews should build player, team, and glossary documents offline."""
        docs = build_seed_documents(
            lineups=[{
                "team_name": "Argentina",
                "lineup": [{"player_name": "Lionel Messi"}],
            }],
            meta={
                "home_team": {"home_team_name": "Argentina"},
                "away_team": {"away_team_name": "France"},
            },
        )
        status = seed_context_store(docs)

        self.assertGreaterEqual(len(docs), 4)
        self.assertIn("Lionel Messi", {doc["name"] for doc in docs})
        self.assertEqual(status["ready"], False)
        self.assertEqual(status["pending"], len(docs))


class Day4PlaceholderTests(unittest.TestCase):
    """Verify the TTS and orchestration placeholders are wired safely."""

    def test_noop_speaker_returns_text_without_audio(self) -> None:
        """The default speaker should never require credentials or write files."""
        result = NoOpSpeaker().synthesize("Goal for Argentina", language="en-US")

        self.assertEqual(result.text, "Goal for Argentina")
        self.assertEqual(result.language, "en-US")
        self.assertFalse(result.has_audio())

    def test_stream_commentary_connects_replay_agent_context_and_tts(self) -> None:
        """The placeholder pipeline should emit text plus a no-op speech result."""
        outputs = list(stream_commentary(
            [shot_event(1, "Goal")],
            language="en",
            speed=0,
            mock=True,
            context_enabled=True,
            tts_enabled=True,
        ))

        self.assertEqual(len(outputs), 1)
        self.assertIn("GOAL", outputs[0].text)
        self.assertEqual(outputs[0].speech.language, "en-US")
        self.assertFalse(outputs[0].speech.has_audio())
        self.assertEqual(outputs[0].as_dict()["event_type"], "Shot")
