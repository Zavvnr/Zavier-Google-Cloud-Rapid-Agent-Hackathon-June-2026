"""
agent/dead_air.py  —  SCAFFOLD (Feature 1: lull / dead-air color commentary).

Three small, decoupled pieces so you can wire them into your agent loop without me
touching your existing files:

  * LiveTallies      — running per-player counts from the event stream (the "wow"
                       layer: "Messi already with 2 key passes").
  * LullDetector     — the inverse of pacing: detects quiet stretches worth filling.
  * ColorCommentator — assembles a color line about the player on the ball, using
                       retrieved player context (MCP) + live tallies, and ROTATES
                       the angle so it never repeats for the same player.

STATUS: the timing / tally / angle-rotation logic is implemented and
offline-testable. The actual Gemini text call is stubbed (mock mode) for you to
wire to your client.

Wiring sketch (in your per-event loop):

    tallies.observe(ev)                                   # every event
    lull.observe_importance(state_seconds, importance(ev))
    if not agent.should_comment(ev) and lull.is_lull(state_seconds):
        line = color.comment(ev, state_dict, context_client, tallies)
        if line:
            lull.note_comment(state_seconds)
            emit(line)                                     # analyst voice (Feature 2)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# Angles the analyst rotates through so color never repeats for a player.
ANGLES = ("tournament_form", "club_role", "national_importance", "tactical_phase")


def _name(value) -> str:
    return (value or {}).get("name", "") if isinstance(value, dict) else ""


# --------------------------------------------------------------------------- #
# Live in-match tallies                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class LiveTallies:
    """Per-player running counts computed from the stream so far (pure)."""

    counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def observe(self, ev: dict) -> None:
        """Update tallies from one event. Call for EVERY event."""
        player = _name(ev.get("player"))
        if not player:
            return
        c = self.counts[player]
        c["touches"] += 1
        etype = _name(ev.get("type"))
        if etype == "Pass":
            c["passes"] += 1
            if (ev.get("pass") or {}).get("shot_assist"):
                c["key_passes"] += 1
        elif etype == "Shot":
            c["shots"] += 1
            if _name((ev.get("shot") or {}).get("outcome")) == "Goal":
                c["goals"] += 1
        elif etype == "Dribble":
            if _name((ev.get("dribble") or {}).get("outcome")) == "Complete":
                c["dribbles"] += 1

    def summary(self, player: str) -> str:
        """Short faithful summary, e.g. 'Messi: 2 key passes, 1 shot, 18 touches'."""
        c = self.counts.get(player)
        if not c:
            return ""
        bits = []
        labels = (
            ("goal", "goals"),
            ("key pass", "key_passes"),
            ("shot", "shots"),
            ("completed dribble", "dribbles"),
        )
        for singular, key in labels:
            value = c.get(key, 0)
            if value:
                label = singular if value == 1 else f"{singular}s"
                bits.append(f"{value} {label}")
        touches = c.get("touches", 0)
        bits.append(f"{touches} {'touch' if touches == 1 else 'touches'}")
        return f"{player}: " + ", ".join(bits)


# --------------------------------------------------------------------------- #
# Lull detection (inverse pacing)                                             #
# --------------------------------------------------------------------------- #
@dataclass
class LullDetector:
    """
    Detects fillable quiet stretches. The caller feeds match-seconds and the
    importance of each event; a lull is when nothing notable has happened for
    `lull_after_s` AND we haven't filled within `color_cooldown_s` (so it never
    becomes a trivia firehose).
    """

    lull_after_s: int = 18          # quiet gap before color kicks in
    color_cooldown_s: int = 25      # minimum spacing between color lines
    notable_importance: float = 0.5
    _first_event_s: Optional[int] = None
    _last_notable_s: Optional[int] = None
    _last_color_s: Optional[int] = None

    def observe_importance(self, seconds: int, importance: float) -> None:
        """Record that an event of this importance happened at `seconds`."""
        if self._first_event_s is None:
            self._first_event_s = seconds
        if importance >= self.notable_importance:
            self._last_notable_s = seconds

    def is_lull(self, seconds: int) -> bool:
        """True when it's quiet enough (and we're off cooldown) to fill the gap."""
        baseline = self._last_notable_s
        if baseline is None:
            baseline = self._first_event_s
        if baseline is None:
            return False
        if seconds - baseline < self.lull_after_s:
            return False
        if self._last_color_s is not None and seconds - self._last_color_s < self.color_cooldown_s:
            return False
        return True

    def note_comment(self, seconds: int) -> None:
        """Reset the color cooldown after emitting a color line."""
        self._last_color_s = seconds


# --------------------------------------------------------------------------- #
# Color commentator (analyst voice for Feature 2)                             #
# --------------------------------------------------------------------------- #
@dataclass
class ColorCommentator:
    """
    Builds a color line about the player on the ball, rotating the angle per player
    so it never repeats. Faithful by construction: it only forwards retrieved facts
    + live tallies to the model and instructs "no invented stats".

    `generate` is an injected callable(prompt)->str (your Gemini client). In mock
    mode (or if `generate` is None) it returns a deterministic stub so the loop is
    testable offline.
    """

    language: str = "en-US"
    generate: Optional[Callable[[str], str]] = None
    mock: bool = False
    _angle_idx: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _next_angle(self, player: str) -> str:
        i = self._angle_idx[player]
        self._angle_idx[player] = i + 1
        return ANGLES[i % len(ANGLES)]

    def build_prompt(self, player, angle, context, tally_summary, state) -> str:
        """Assemble the color prompt. TODO: blend with your prompts package."""
        return (
            f"Color commentary in {self.language}. Player on the ball: {player}. "
            f"Angle to take: {angle}. Retrieved facts: {context or {}}. "
            f"Live tally so far: {tally_summary or 'n/a'}. Match state: {state or {}}. "
            "Write ONE faithful sentence. Use only the facts/tallies given — never "
            "invent stats. Tie it to the current phase, not a generic bio dump."
        )

    def comment(
        self,
        ev,
        state,
        context_client=None,
        tallies: Optional[LiveTallies] = None,
        context: Optional[dict] = None,
    ) -> Optional[str]:
        """Produce a color line for the player on the ball, or None."""
        player = _name(ev.get("player"))
        if not player:
            return None
        angle = self._next_angle(player)

        if context is None:
            context = {}
        if not context and context_client is not None:
            try:
                context = context_client.fetch_event_context(ev, state)
            except Exception:
                context = {}
        tally_summary = tallies.summary(player) if tallies else ""

        if self.mock or self.generate is None:
            return f"[{self.language}|color/{angle}] {tally_summary or player} — analyst color (stub)."

        prompt = self.build_prompt(player, angle, context, tally_summary, state)
        try:
            text = (self.generate(prompt) or "").strip()
        except Exception:
            return None
        return text or None
