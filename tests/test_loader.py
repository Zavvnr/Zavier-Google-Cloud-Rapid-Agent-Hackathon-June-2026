"""Unit tests for the StatsBomb loader/cache layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import loader


class LoaderCacheTests(unittest.TestCase):
    """Verify that match data is cached in the expected on-disk layout."""

    def test_cache_match_writes_events_lineups_meta_and_attribution(self) -> None:
        """Cache one match using mocked downloads and assert each file is written."""
        events = [{"id": "event-1", "type": {"name": "Shot"}}]
        lineups = [{"team_name": "Argentina", "lineup": []}]
        meta = {"match_id": 123, "home_team": {"home_team_name": "Argentina"}}

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            with patch.object(loader, "CACHE_DIR", cache_root), \
                    patch.object(loader, "download_events", return_value=events), \
                    patch.object(loader, "download_lineups", return_value=lineups), \
                    patch.object(loader, "find_match_meta", return_value=meta):
                dest = loader.cache_match(123)

            self.assertEqual(dest, cache_root / "123")
            self.assertEqual(json.loads((dest / "events.json").read_text()), events)
            self.assertEqual(json.loads((dest / "lineups.json").read_text()), lineups)
            self.assertEqual(json.loads((dest / "meta.json").read_text()), meta)
            self.assertIn("StatsBomb", (dest / "ATTRIBUTION.txt").read_text())

    def test_load_cached_events_raises_helpful_error_when_missing(self) -> None:
        """A missing cache should tell the caller how to download the match."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(loader, "CACHE_DIR", Path(tmp)):
                with self.assertRaisesRegex(FileNotFoundError, "python data/loader.py"):
                    loader.load_cached_events(999)
