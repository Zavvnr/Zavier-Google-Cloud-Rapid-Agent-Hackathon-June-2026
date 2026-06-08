"""
Offline tests for the Feature 1 / Feature 2 SCAFFOLDS.

Covers only the pure, deterministic logic (aggregation, tallies, lull timing,
turn-taking, dialogue parsing, TTS ordering/fallback) — the parts that are real
even though the Gemini/TTS calls are stubbed. No network, keys, or DB needed.

    python -m unittest tests.test_feature_scaffolds -v
"""
from __future__ import annotations

import unittest

from context.player_stats import aggregate_player_form, build_player_documents
from agent.dead_air import LiveTallies, LullDetector, ColorCommentator
from agent.commentary_crew import (
    CommentaryCrew, TurnTakingController, Turn, parse_dialogue,
)
from tts.multispeaker import MultiSpeakerSpeaker


# --- helpers ---------------------------------------------------------------- #
def _pass(player, team, shot_assist=False, goal_assist=False):
    p = {"type": {"name": "Pass"}, "team": {"name": team}, "player": {"name": player},
         "pass": {}}
    if shot_assist:
        p["pass"]["shot_assist"] = True
    if goal_assist:
        p["pass"]["goal_assist"] = True
    return p


def _shot(player, team, goal=False):
    return {"type": {"name": "Shot"}, "team": {"name": team}, "player": {"name": player},
            "shot": {"outcome": {"name": "Goal" if goal else "Saved"}}}


# --- Feature 1: player stats ------------------------------------------------ #
class PlayerStatsTests(unittest.TestCase):
    def test_aggregate_counts_across_matches(self):
        events_by_match = {
            "A": [_pass("Messi", "Argentina", shot_assist=True),
                  _shot("Messi", "Argentina", goal=True),
                  _pass("Messi", "Argentina", goal_assist=True),
                  _shot("Mbappe", "France", goal=False)],
            "B": [_pass("Messi", "Argentina")],
        }
        forms = aggregate_player_form(events_by_match)
        messi = forms["Messi"]
        self.assertEqual(messi.matches, 2)
        self.assertEqual(messi.goals, 1)
        self.assertEqual(messi.assists, 1)
        self.assertEqual(messi.key_passes, 1)
        self.assertEqual(messi.shots, 1)
        self.assertEqual(messi.passes, 3)
        self.assertEqual(messi.team, "Argentina")
        self.assertEqual(forms["Mbappe"].matches, 1)

    def test_aggregate_skips_missing_players_and_counts_one_appearance_per_match(self):
        events_by_match = {
            "A": [
                {"type": {"name": "Half Start"}, "team": {"name": "Argentina"}},
                _pass("Messi", "Argentina"),
                _pass("Messi", "Argentina"),
            ],
        }
        forms = aggregate_player_form(events_by_match)
        self.assertEqual(set(forms), {"Messi"})
        self.assertEqual(forms["Messi"].matches, 1)
        self.assertEqual(forms["Messi"].passes, 2)

    def test_build_documents_are_player_kind_with_stats(self):
        forms = aggregate_player_form({"A": [_shot("Messi", "Argentina", goal=True)]})
        docs = build_player_documents(forms, bios={"Messi": "GOAT."})
        self.assertEqual(docs[0]["kind"], "player")
        self.assertIn("1 goal", docs[0]["text"])
        self.assertIn("GOAT.", docs[0]["text"])


# --- Feature 1: live tallies + lull ---------------------------------------- #
class DeadAirTests(unittest.TestCase):
    def test_live_tallies_track_key_passes_and_shots(self):
        tallies = LiveTallies()
        tallies.observe(_pass("Messi", "Argentina", shot_assist=True))
        tallies.observe(_shot("Messi", "Argentina", goal=False))
        summary = tallies.summary("Messi")
        self.assertIn("1 key pass", summary)
        self.assertIn("1 shot", summary)
        self.assertIn("touches", summary)

    def test_lull_detector_respects_gap_and_cooldown(self):
        lull = LullDetector(lull_after_s=18, color_cooldown_s=25)
        lull.observe_importance(0, 1.0)            # notable at t=0
        self.assertFalse(lull.is_lull(10))         # only 10s quiet
        self.assertTrue(lull.is_lull(20))          # 20s quiet -> fillable
        lull.note_comment(20)
        self.assertFalse(lull.is_lull(30))         # still on cooldown
        self.assertTrue(lull.is_lull(50))          # cooldown elapsed

    def test_lull_detector_can_fill_opening_quiet_stretch(self):
        lull = LullDetector(lull_after_s=10, color_cooldown_s=25)
        lull.observe_importance(0, 0.1)
        self.assertFalse(lull.is_lull(9))
        self.assertTrue(lull.is_lull(10))

    def test_color_commentator_rotates_angle_and_is_offline_safe(self):
        color = ColorCommentator(language="es-ES", mock=True)
        ev = _pass("Messi", "Argentina")
        first = color.comment(ev, {"clock": "22:10"})
        second = color.comment(ev, {"clock": "22:30"})
        self.assertIn("color/tournament_form", first)
        self.assertIn("color/club_role", second)   # angle advanced -> no repetition


# --- Feature 2: turn-taking + dialogue ------------------------------------- #
class CrewTests(unittest.TestCase):
    def test_turn_taking_plan(self):
        c = TurnTakingController()
        goal = c.plan(_shot("Messi", "Argentina", goal=True), 1.0, False)
        self.assertEqual((goal.kind, goal.speakers), ("goal", ["lead", "analyst"]))
        call = c.plan({"type": {"name": "Foul Committed"}}, 0.7, False)
        self.assertEqual((call.kind, call.speakers), ("call", ["lead"]))
        color = c.plan({"type": {"name": "Pass"}}, 0.1, True)
        self.assertEqual((color.kind, color.speakers), ("color", ["analyst"]))
        self.assertIsNone(c.plan({"type": {"name": "Pass"}}, 0.1, False))

    def test_parse_dialogue(self):
        script = parse_dialogue("Lead: GOAL!\nAnalyst: What a finish.")
        self.assertEqual([t.speaker for t in script.turns], ["lead", "analyst"])
        self.assertEqual(script.turns[0].text, "GOAL!")

    def test_parse_dialogue_records_audio_tags_and_ignores_unlabeled_lines(self):
        script = parse_dialogue("Noise\nLead: [excited] GOAL!\nCoach: no\nAnalyst: Calm read.")
        self.assertEqual([t.speaker for t in script.turns], ["lead", "analyst"])
        self.assertEqual(script.turns[0].audio_tags, ["excited"])

    def test_crew_mock_script_for_goal_has_both_speakers(self):
        crew = CommentaryCrew(language="es-ES", mock=True)
        plan = TurnTakingController().plan(_shot("Messi", "Argentina", goal=True), 1.0, False)
        script = crew.generate_script(_shot("Messi", "Argentina", goal=True), {}, plan)
        self.assertEqual([t.speaker for t in script.turns], ["lead", "analyst"])
        self.assertIn("excited", script.turns[0].audio_tags)

    def test_crew_filters_speakers_not_in_plan(self):
        crew = CommentaryCrew(generate=lambda prompt: "Lead: Fine pass.\nAnalyst: Extra.")
        plan = TurnTakingController().plan({"type": {"name": "Pass"}}, 0.7, False)
        script = crew.generate_script({"type": {"name": "Pass"}}, {}, plan)
        self.assertEqual([t.speaker for t in script.turns], ["lead"])

    def test_goal_script_keeps_goal_call_to_lead_only(self):
        crew = CommentaryCrew(
            generate=lambda prompt: "Lead: [excited] GOAL!\nAnalyst: GOAL! What a finish."
        )
        plan = TurnTakingController().plan(_shot("Messi", "Argentina", goal=True), 1.0, False)
        script = crew.generate_script(_shot("Messi", "Argentina", goal=True), {}, plan)
        self.assertEqual(script.turns[0].speaker, "lead")
        self.assertIn("GOAL", script.turns[0].text)
        self.assertEqual(script.turns[1].speaker, "analyst")
        self.assertEqual(script.turns[1].text, "What a finish.")


# --- Feature 2: multi-speaker TTS ------------------------------------------ #
class _FakeSpeaker:
    def synthesize(self, text, language="en-US"):
        class _R:
            audio_bytes = b"AUDIO"
            skipped_reason = ""
        return _R()


class MultiSpeakerTests(unittest.TestCase):
    TURNS = [Turn("lead", "GOAL!", ["excited"]), Turn("analyst", "Brilliant.")]

    def test_mock_returns_sequential_placeholder_audio(self):
        audio = MultiSpeakerSpeaker(mock=True).synthesize_dialogue(self.TURNS)
        self.assertEqual([s.speaker for s in audio.segments], ["lead", "analyst"])
        self.assertTrue(audio.has_audio())

    def test_path_b_uses_injected_single_speaker(self):
        spk = MultiSpeakerSpeaker(path="B", single_speaker=_FakeSpeaker())
        audio = spk.synthesize_dialogue(self.TURNS)
        self.assertTrue(all(s.audio_bytes == b"AUDIO" for s in audio.segments))

    def test_path_a_uses_injected_transport(self):
        spk = MultiSpeakerSpeaker(path="A",
                                  multispeaker_transport=lambda turns, lang: [b"A1", b"A2"])
        audio = spk.synthesize_dialogue(self.TURNS)
        self.assertEqual([s.audio_bytes for s in audio.segments], [b"A1", b"A2"])

    def test_path_a_falls_back_to_b_when_transport_missing(self):
        spk = MultiSpeakerSpeaker(path="A", single_speaker=_FakeSpeaker())  # no transport
        audio = spk.synthesize_dialogue(self.TURNS)
        self.assertTrue(audio.has_audio())  # fell back to Path B

    def test_path_a_falls_back_to_b_when_segment_count_is_wrong(self):
        spk = MultiSpeakerSpeaker(path="A",
                                  multispeaker_transport=lambda turns, lang: [b"ONLY_ONE"],
                                  single_speaker=_FakeSpeaker())
        audio = spk.synthesize_dialogue(self.TURNS)
        self.assertEqual(len(audio.segments), 2)
        self.assertTrue(all(s.audio_bytes == b"AUDIO" for s in audio.segments))


if __name__ == "__main__":
    unittest.main()
