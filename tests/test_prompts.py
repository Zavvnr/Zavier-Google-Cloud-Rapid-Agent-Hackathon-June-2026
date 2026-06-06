"""Unit tests for Day 2 prompt and language helpers."""

from __future__ import annotations

import unittest

from agent import prompts


class PromptTests(unittest.TestCase):
    """Verify language normalization and faithful event prompt assembly."""

    def test_short_language_aliases_normalize_to_supported_codes(self) -> None:
        """Short demo codes should work alongside the full Text-to-Speech codes."""
        self.assertEqual(prompts.normalize_language("en"), "en-US")
        self.assertEqual(prompts.normalize_language("es"), "es-ES")
        self.assertEqual(prompts.normalize_language("id"), "id-ID")
        self.assertIn("en", prompts.SUPPORTED_LANGUAGE_CODES)
        self.assertIn("en-US", prompts.SUPPORTED_LANGUAGE_CODES)

    def test_system_prompt_uses_language_display_name(self) -> None:
        """The composed system prompt should expand aliases into readable names."""
        system = prompts.system_prompt("es")
        self.assertIn("Spanish (Spain)", system)
        self.assertIn("Your goal is", system)

    def test_build_event_prompt_includes_only_current_event_facts(self) -> None:
        """The user prompt should carry match state plus factual event details."""
        event = {
            "type": {"name": "Shot"},
            "team": {"name": "Argentina"},
            "player": {"name": "Lionel Messi"},
            "shot": {
                "outcome": {"name": "Saved"},
                "body_part": {"name": "Left Foot"},
                "statsbomb_xg": 0.42,
            },
        }
        prompt = prompts.build_event_prompt(event, {"score": "Argentina 0-0 France"})

        self.assertIn("MATCH STATE", prompt)
        self.assertIn("type=Shot", prompt)
        self.assertIn("player=Lionel Messi", prompt)
        self.assertIn("outcome=Saved", prompt)
        self.assertIn("xg=0.42", prompt)
