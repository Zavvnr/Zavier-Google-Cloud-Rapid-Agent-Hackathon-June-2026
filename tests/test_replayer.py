"""Unit tests for the Day 2 event replayer."""

from __future__ import annotations

import unittest

from replayer.event_replayer import parse_timestamp, replay, summarize_event


class ReplayerTests(unittest.TestCase):
    """Verify timestamp parsing, ordering, and accelerated waits."""

    def test_parse_timestamp_returns_period_elapsed_seconds(self) -> None:
        """StatsBomb timestamps should become seconds within the current period."""
        self.assertEqual(parse_timestamp("00:01:02.500"), 62.5)
        self.assertEqual(parse_timestamp("01:00:00.000"), 3600.0)

    def test_replay_orders_by_index_and_skips_period_gap(self) -> None:
        """Replay should sleep within a period but not across half-time changes."""
        events = [
            {"index": 2, "period": 1, "timestamp": "00:00:04.000", "type": {"name": "Pass"}},
            {"index": 1, "period": 1, "timestamp": "00:00:00.000", "type": {"name": "Pass"}},
            {"index": 3, "period": 2, "timestamp": "00:15:00.000", "type": {"name": "Pass"}},
        ]
        waits: list[float] = []

        ordered = list(replay(events, speed=2.0, sleep=waits.append))

        self.assertEqual([event["index"] for event in ordered], [1, 2, 3])
        self.assertEqual(waits, [2.0])

    def test_summarize_event_includes_clock_type_team_and_player(self) -> None:
        """The CLI summary should be compact but identify the event clearly."""
        summary = summarize_event({
            "period": 1,
            "minute": 22,
            "second": 8,
            "type": {"name": "Shot"},
            "team": {"name": "Argentina"},
            "player": {"name": "Angel Di Maria"},
        })

        self.assertIn("P1 22:08", summary)
        self.assertIn("Shot", summary)
        self.assertIn("Argentina", summary)
        self.assertIn("Angel Di Maria", summary)
