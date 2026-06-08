"""Unit and integration tests for the commentary agent."""

from __future__ import annotations

import unittest

from agent.commentary_agent import CommentaryAgent, importance
from agent.dead_air import LullDetector
from replayer.event_replayer import replay


def pass_event(index: int, minute: int, second: int, end_location: list[float]) -> dict:
    """Build a minimal StatsBomb-shaped pass event for pacing tests."""
    return {
        "index": index,
        "period": 1,
        "timestamp": f"00:{minute:02d}:{second:02d}.000",
        "minute": minute,
        "second": second,
        "type": {"name": "Pass"},
        "team": {"name": "Argentina"},
        "player": {"name": "Lionel Messi"},
        "pass": {"end_location": end_location},
    }


def shot_event(index: int, minute: int, second: int, outcome: str) -> dict:
    """Build a minimal StatsBomb-shaped shot event for score and run tests."""
    return {
        "index": index,
        "period": 1,
        "timestamp": f"00:{minute:02d}:{second:02d}.000",
        "minute": minute,
        "second": second,
        "type": {"name": "Shot"},
        "team": {"name": "Argentina"},
        "player": {"name": "Lionel Messi"},
        "shot": {"outcome": {"name": outcome}},
    }


class CommentaryAgentTests(unittest.TestCase):
    """Verify offline commentary behavior without calling Gemini."""

    def test_importance_scores_big_moments_above_routine_events(self) -> None:
        """Goals should outrank dangerous passes, which should outrank routine play."""
        self.assertEqual(importance(shot_event(1, 1, 0, "Goal")), 1.0)
        self.assertGreater(importance(pass_event(2, 1, 5, [108.0, 40.0])), 0.3)
        self.assertLess(importance(pass_event(3, 1, 8, [60.0, 40.0])), 0.3)

    def test_handle_updates_score_before_mock_goal_line(self) -> None:
        """A generated goal line should include the updated scoreline."""
        agent = CommentaryAgent(
            language="en",
            mock=True,
            home_team="Argentina",
            away_team="France",
        )

        line = agent.handle(shot_event(1, 22, 8, "Goal"))

        self.assertIsNotNone(line)
        self.assertIn("Argentina 1-0 France", line or "")
        self.assertEqual(agent.language, "en-US")

    def test_handle_fetches_context_through_injected_client(self) -> None:
        """The Day 3 context seam should be called before line generation."""
        class FakeContextClient:
            """Small test double that records context fetch calls."""

            def __init__(self) -> None:
                """Start with no recorded context calls."""
                self.calls = []

            def fetch_event_context(self, event: dict, state: dict) -> dict:
                """Record the event/state and return harmless context."""
                self.calls.append((event, state))
                return {"player_note": "captain"}

        fake_context = FakeContextClient()
        agent = CommentaryAgent(language="en", mock=True, context_client=fake_context)

        line = agent.handle(shot_event(1, 22, 8, "Saved"))

        self.assertIsNotNone(line)
        self.assertEqual(len(fake_context.calls), 1)
        self.assertEqual(fake_context.calls[0][1]["clock"], "22:08")

    def test_should_comment_respects_skip_types_and_cooldown(self) -> None:
        """Routine skip events and medium-importance cooldowns should stay quiet."""
        agent = CommentaryAgent(language="en", mock=True)

        skip_line = agent.handle({
            "index": 1,
            "period": 1,
            "timestamp": "00:00:01.000",
            "minute": 0,
            "second": 1,
            "type": {"name": "Ball Receipt*"},
            "team": {"name": "Argentina"},
        })
        first_line = agent.handle(pass_event(2, 1, 0, [108.0, 40.0]))
        cooled_down_line = agent.handle(pass_event(3, 1, 5, [108.0, 40.0]))
        after_cooldown_line = agent.handle(pass_event(4, 1, 13, [108.0, 40.0]))

        self.assertIsNone(skip_line)
        self.assertIsNotNone(first_line)
        self.assertIsNone(cooled_down_line)
        self.assertIsNotNone(after_cooldown_line)

    def test_dead_air_emits_analyst_color_after_quiet_opening(self) -> None:
        """Routine early possession should eventually get analyst color, not silence."""
        agent = CommentaryAgent(
            language="en",
            mock=True,
            lull_detector=LullDetector(lull_after_s=10, color_cooldown_s=30),
        )

        first = agent.handle(pass_event(1, 0, 0, [60.0, 40.0]))
        second = agent.handle_item(pass_event(2, 0, 10, [61.0, 40.0]))

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.kind, "color")
        self.assertEqual(second.speaker, "analyst")
        self.assertIn("color/tournament_form", second.text)

    def test_dead_air_can_be_disabled(self) -> None:
        """The old event-only pacing mode should remain available."""
        agent = CommentaryAgent(
            language="en",
            mock=True,
            dead_air_enabled=False,
            lull_detector=LullDetector(lull_after_s=1, color_cooldown_s=30),
        )

        self.assertIsNone(agent.handle(pass_event(1, 0, 0, [60.0, 40.0])))
        self.assertIsNone(agent.handle(pass_event(2, 0, 5, [61.0, 40.0])))

    def test_two_speaker_goal_turns_are_ordered_lead_then_analyst(self) -> None:
        """Goal moments should be a sequential lead call followed by analyst reaction."""
        agent = CommentaryAgent(language="en", mock=True, two_speakers=True)

        item = agent.handle_item(shot_event(1, 22, 8, "Goal"))

        self.assertIsNotNone(item)
        self.assertEqual(item.kind, "goal")
        self.assertEqual([turn.speaker for turn in item.turns], ["lead", "analyst"])
        self.assertIn("excited", item.turns[0].audio_tags)
        self.assertIn("Lead:", item.text)
        self.assertIn("Analyst:", item.text)

    def test_two_speaker_lull_uses_analyst_color_hint_only(self) -> None:
        """Lull commentary should be analyst-led rather than a lead play call."""
        agent = CommentaryAgent(
            language="en",
            mock=True,
            two_speakers=True,
            lull_detector=LullDetector(lull_after_s=5, color_cooldown_s=30),
        )

        self.assertIsNone(agent.handle_item(pass_event(1, 0, 0, [60.0, 40.0])))
        item = agent.handle_item(pass_event(2, 0, 5, [61.0, 40.0]))

        self.assertIsNotNone(item)
        self.assertEqual(item.kind, "color")
        self.assertEqual([turn.speaker for turn in item.turns], ["analyst"])
        self.assertIn("color/tournament_form", item.turns[0].text)

    def test_replayer_to_agent_mock_pipeline_yields_commentary_lines(self) -> None:
        """The Day 2 integration path should stream replayed events into the agent."""
        events = [
            pass_event(1, 1, 0, [108.0, 40.0]),
            shot_event(2, 1, 20, "Goal"),
        ]
        agent = CommentaryAgent(
            language="id",
            mock=True,
            home_team="Argentina",
            away_team="France",
        )

        lines = list(agent.run(replay(events, speed=0)))

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][0]["type"]["name"], "Pass")
        self.assertIn("[id-ID|01']", lines[1][1])
        self.assertIn("Argentina 1-0 France", lines[1][1])
