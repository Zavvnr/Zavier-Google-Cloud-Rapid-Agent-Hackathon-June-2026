"""
agent/commentary_crew.py  —  SCAFFOLD (Feature 2: lead + analyst two-speaker layer).

A turn-taking controller + a two-speaker dialogue generator. The exchange is
produced as ONE labeled script in a SINGLE Gemini text call, which keeps the two
voices coherent and lets them reference each other. The ANALYST turn is the
delivery vehicle for Feature 1's dead-air color commentary.

Goal-moment handling (important): the lead delivers the energetic call (with TTS
audio tags), THEN the analyst reacts — SEQUENTIAL, never overlapping. Synthesised
voices over each other are unintelligible and aren't how a real goal sounds (it's
the lead's crescendo, then the analyst). Audio ordering lives in tts/multispeaker.py.

STATUS: turn-taking + script parsing are implemented + offline-testable; the Gemini
call is stubbed (mock) for you to wire.

Wiring sketch (per event):
    plan = controller.plan(ev, importance(ev), lull.is_lull(seconds))
    if plan:
        script = crew.generate_script(ev, state, plan, context, color_hint)
        audio  = multispeaker.synthesize_dialogue(script.turns)   # sequential
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

LEAD, ANALYST = "lead", "analyst"


@dataclass
class Turn:
    """One speaker's line, with optional Gemini-TTS audio tags (not SSML)."""

    speaker: str            # "lead" | "analyst"
    text: str
    audio_tags: List[str] = field(default_factory=list)


@dataclass
class DialogueScript:
    """An ordered lead/analyst exchange."""

    turns: List[Turn] = field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(f"{t.speaker.capitalize()}: {t.text}" for t in self.turns)


@dataclass
class TurnPlan:
    """Who should speak for this moment, and what kind of moment it is."""

    kind: str               # "call" | "goal" | "color"
    speakers: List[str]


def _default_is_goal(ev: dict) -> bool:
    """Goal iff a Shot with outcome Goal, or an own-goal event type."""
    etype = (ev.get("type") or {}).get("name")
    if etype == "Shot":
        return ((ev.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
    return etype in ("Own Goal For", "Own Goal Against")


class TurnTakingController:
    """Decide who speaks for an event (the agentic layer). Pure + testable."""

    def __init__(self, is_goal: Optional[Callable[[dict], bool]] = None,
                 call_importance: float = 0.65):
        self.is_goal = is_goal or _default_is_goal
        self.call_importance = call_importance

    def plan(self, ev: dict, importance: float, is_lull: bool) -> Optional[TurnPlan]:
        """Return a TurnPlan, or None to stay silent."""
        if self.is_goal(ev):
            return TurnPlan("goal", [LEAD, ANALYST])     # lead's big call, then analyst
        if importance >= self.call_importance:
            return TurnPlan("call", [LEAD])              # lead play-by-play
        if is_lull:
            return TurnPlan("color", [ANALYST])          # analyst-led dead-air color
        return None                                       # quiet


_LABEL_RE = re.compile(r"^\s*(lead|analyst)\s*:\s*(.+)$", re.IGNORECASE)
_AUDIO_TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
_LEADING_GOAL_CALL_RE = re.compile(
    r"^(\s*(?:\[[^\]]+\]\s*)*)(?:[!?.]\s*)*"
    r"(?:g+o+a+l+|g+o+l+|goalazo|but|tor)\b[\w!?.:-]*\s*",
    re.IGNORECASE,
)


def parse_dialogue(raw: str) -> DialogueScript:
    """Parse a 'Lead: ...\\nAnalyst: ...' labeled script into turns. Pure + testable."""
    turns: List[Turn] = []
    for line in (raw or "").splitlines():
        m = _LABEL_RE.match(line)
        if m:
            text = m.group(2).strip()
            turns.append(
                Turn(
                    speaker=m.group(1).lower(),
                    text=text,
                    audio_tags=_AUDIO_TAG_RE.findall(text),
                )
            )
    return DialogueScript(turns=turns)


def _remove_analyst_goal_call(text: str) -> str:
    """Ensure the analyst reaction does not become a second goal shout."""
    cleaned = _LEADING_GOAL_CALL_RE.sub(r"\1", text or "").strip()
    if cleaned and cleaned != text.strip():
        if _AUDIO_TAG_RE.sub("", cleaned).strip():
            return cleaned
        return "What a finish from the lead call."
    if _LEADING_GOAL_CALL_RE.fullmatch(text or ""):
        return "What a finish from the lead call."
    return (text or "").strip()


@dataclass
class CommentaryCrew:
    """
    Generate the two-speaker exchange in ONE Gemini text call (stubbed in mock).

    `generate` is an injected callable(prompt)->str (your Gemini client). For goals,
    the prompt asks for TTS audio tags on the lead's call. The analyst turn can be
    fed a `color_hint` from Feature 1's ColorCommentator so the two features compose.
    """

    language: str = "en-US"
    generate: Optional[Callable[[str], str]] = None
    mock: bool = False

    def build_prompt(self, ev, state, plan: TurnPlan, context, color_hint="") -> str:
        """Assemble the two-speaker prompt. TODO: blend with your prompts package."""
        tag_hint = (
            " For goals: only Lead may say the goal call ('GOAL', 'Gol', etc.). "
            "Analyst speaks after Lead, reacts without repeating the goal shout, "
            "and never overlaps. Put Gemini-TTS audio tags (e.g. [excited]) on "
            "the lead's goal call."
            if plan.kind == "goal" else ""
        )
        return (
            f"Two-speaker football commentary in {self.language}. Moment: {plan.kind}. "
            f"Speakers: {', '.join(plan.speakers)}. Event: {(ev.get('type') or {}).get('name')}. "
            f"Match state: {state or {}}. Retrieved context: {context or {}}. "
            f"Analyst color hint: {color_hint or 'n/a'}. "
            "Write a SHORT labeled script with 'Lead:' and/or 'Analyst:' lines, "
            "faithful to the data, no invented facts." + tag_hint
        )

    def generate_script(self, ev, state, plan: TurnPlan, context=None, color_hint="") -> DialogueScript:
        """Return the lead/analyst exchange for this moment."""
        if self.mock or self.generate is None:
            return self._mock_script(ev, plan, color_hint=color_hint)
        prompt = self.build_prompt(ev, state, plan, context or {}, color_hint)
        try:
            raw = self.generate(prompt) or ""
        except Exception:
            return DialogueScript(turns=[])
        script = parse_dialogue(raw)
        allowed = set(plan.speakers)
        script.turns = [turn for turn in script.turns if turn.speaker in allowed]
        if plan.kind == "goal":
            for turn in script.turns:
                if turn.speaker == ANALYST:
                    turn.text = _remove_analyst_goal_call(turn.text)
        return script

    def _mock_script(self, ev, plan: TurnPlan, color_hint: str = "") -> DialogueScript:
        """Deterministic offline script so the controller is testable now."""
        etype = (ev.get("type") or {}).get("name", "?")
        turns: List[Turn] = []
        if LEAD in plan.speakers:
            tags = ["excited"] if plan.kind == "goal" else []
            if plan.kind == "goal":
                player = ((ev.get("player") or {}).get("name") or "the scorer").split(" ")[-1]
                team = (ev.get("team") or {}).get("name", "")
                text = f"[{self.language}] GOAL! {player} scores for {team}! (stub)."
            else:
                text = f"[{self.language}] Lead call: {etype} (stub)."
            turns.append(Turn(LEAD, text, tags))
        if ANALYST in plan.speakers:
            text = color_hint or f"[{self.language}] Analyst reaction/color (stub)."
            turns.append(Turn(ANALYST, text))
        return DialogueScript(turns=turns)
