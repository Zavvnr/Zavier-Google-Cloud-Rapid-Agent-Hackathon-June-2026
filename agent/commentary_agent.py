"""
agent/commentary_agent.py  —  Day 2 deliverable.

The core generation loop (text only): consume an event stream, decide what's worth
saying (pacing), track the score, and generate each commentary line natively in the
target language — with a hard "no inventing events" guardrail (see agent/prompts).

What's intentionally NOT here yet:
  * MCP context retrieval  -> Day 3 (see `fetch_context`, a stub for now).
  * Text-to-speech          -> Day 4.

CLI (ties the replayer + agent together — the Day 2 end-to-end demo):
    python -m agent.commentary_agent --sample --language en --mock
    python -m agent.commentary_agent --sample --language es           # needs GOOGLE_API_KEY
    python -m agent.commentary_agent --match-id 3869685 --language id --speed 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from agent.mcp_client import NoOpContextClient
from agent import prompts

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Pacing configuration                                                        #
# --------------------------------------------------------------------------- #
# Never comment on these — they're the connective tissue of play, not moments.
SKIP_TYPES = {
    "Ball Receipt*", "Pressure", "Carry", "Ball Recovery", "Miscontrol",
    "Dribbled Past", "Duel", "Dispossessed", "Block", "Half Start",
    "Starting XI", "Tactical Shift", "Camera On", "Camera off", "Player On",
    "Player Off", "Referee Ball-Drop",
}

# An event this important always gets a line, regardless of the cooldown.
HIGH_IMPORTANCE = 0.65
# Below this, never bother — pure filler.
MIN_IMPORTANCE = 0.30
# Stay quiet for at least this many match-seconds between low/medium lines.
COMMENT_COOLDOWN_S = 12


def _in_box(loc) -> bool:
    """StatsBomb pitch is 120x80; the attacking penalty area is x>=102, 18<=y<=62."""
    return bool(loc) and len(loc) >= 2 and loc[0] >= 102 and 18 <= loc[1] <= 62


def importance(ev: dict) -> float:
    """How noteworthy is this event, in [0, 1]?"""
    etype = ev.get("type", {}).get("name", "")

    if etype == "Shot":
        outcome = (ev.get("shot", {}).get("outcome") or {}).get("name", "")
        if outcome == "Goal":
            return 1.0
        if outcome in ("Saved", "Post"):
            return 0.8
        return 0.7  # off target / blocked / wayward — still a chance
    if etype in ("Own Goal For", "Own Goal Against"):
        return 1.0
    if etype == "Goal Keeper":
        gk = (ev.get("goalkeeper", {}).get("type") or {}).get("name", "")
        return 0.7 if "Save" in gk or "Smother" in gk else 0.2
    if etype == "Foul Committed":
        card = (ev.get("foul_committed", {}) or {}).get("card")
        return 0.85 if card else 0.4
    if etype == "Bad Behaviour":
        return 0.85 if (ev.get("bad_behaviour", {}) or {}).get("card") else 0.2
    if etype == "Offside":
        return 0.6
    if etype == "Substitution":
        return 0.5
    if etype == "Foul Won":
        return 0.3
    if etype == "Half End":
        return 0.8
    if etype == "Pass":
        p = ev.get("pass", {})
        if (p.get("technique") or {}).get("name") == "Through Ball":
            return 0.5
        if p.get("cross"):
            return 0.45
        if p.get("shot_assist") or p.get("goal_assist"):
            return 0.6
        if _in_box(p.get("end_location")):
            return 0.45
        return 0.1
    if etype == "Dribble":
        complete = (ev.get("dribble", {}).get("outcome") or {}).get("name") == "Complete"
        return 0.35 if complete and _in_box(ev.get("location")) else 0.15
    return 0.1


# --------------------------------------------------------------------------- #
# Match state                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class MatchState:
    home_team: str = ""
    away_team: str = ""
    score: dict = field(default_factory=dict)
    period: int = 1
    minute: int = 0
    second: int = 0
    last_comment_s: Optional[int] = None
    recent_lines: deque = field(default_factory=lambda: deque(maxlen=4))

    def match_seconds(self) -> int:
        """Return the current match clock as total seconds for pacing math."""
        # StatsBomb `minute` is continuous across halves, so this is monotonic.
        return self.minute * 60 + self.second

    def scoreline(self) -> str:
        """Return a display-ready scoreline from the tracked match state."""
        if not self.score:
            return "0-0"
        h = self.score.get(self.home_team, 0)
        a = self.score.get(self.away_team, 0)
        return f"{self.home_team} {h}-{a} {self.away_team}"

    def as_prompt_dict(self) -> dict:
        """Serialize the state fields that are safe to send to the model."""
        return {
            "clock": f"{self.minute:02d}:{self.second:02d}",
            "period": self.period,
            "score": self.scoreline(),
            "recent_lines": list(self.recent_lines),
        }


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class CommentaryAgent:
    """Stateful, per-event commentary generator with pacing + score tracking."""

    def __init__(
        self,
        language: str = "en",
        model: Optional[str] = None,
        mock: bool = False,
        client=None,
        context_client=None,
        home_team: str = "",
        away_team: str = "",
    ):
        """Create an agent with optional mock mode and an injectable Gemini client."""
        self.language = prompts.normalize_language(language)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-pro")
        self.mock = mock
        self._client = client
        self.context_client = context_client or NoOpContextClient()
        self.state = MatchState(home_team=home_team, away_team=away_team)
        self._system = prompts.system_prompt(self.language)

    # -- match-state bookkeeping -------------------------------------------- #
    def _advance_clock(self, ev: dict) -> None:
        """Move the tracked clock/team names forward using the incoming event."""
        self.state.period = ev.get("period", self.state.period)
        self.state.minute = ev.get("minute", self.state.minute)
        self.state.second = ev.get("second", self.state.second)
        # Infer team names from the stream if not supplied up front.
        team = ev.get("team", {}).get("name")
        if team and not self.state.home_team:
            self.state.home_team = team
        elif team and team != self.state.home_team and not self.state.away_team:
            self.state.away_team = team

    def _apply_score(self, ev: dict) -> None:
        """Update the tracked score for goals explicitly present in the event."""
        etype = ev.get("type", {}).get("name", "")
        team = ev.get("team", {}).get("name", "")
        if etype == "Shot" and (ev.get("shot", {}).get("outcome") or {}).get("name") == "Goal":
            self.state.score[team] = self.state.score.get(team, 0) + 1
        elif etype == "Own Goal For":
            self.state.score[team] = self.state.score.get(team, 0) + 1

    def should_comment(self, ev: dict) -> bool:
        """Decide whether an event is important enough to generate commentary."""
        etype = ev.get("type", {}).get("name", "")
        if etype in SKIP_TYPES:
            return False
        imp = importance(ev)
        if imp >= HIGH_IMPORTANCE:
            return True
        if imp < MIN_IMPORTANCE:
            return False
        last = self.state.last_comment_s
        if last is None:
            return True
        return (self.state.match_seconds() - last) >= COMMENT_COOLDOWN_S

    # -- Day 3 hook: MCP context retrieval (stub for now) ------------------- #
    def fetch_context(self, ev: dict) -> dict:
        """
        Fetch optional Day 3 context for the current event through the MCP seam.

        The default context client returns {}, so Day 2 runs without MongoDB.
        """
        try:
            return self.context_client.fetch_event_context(ev, self.state.as_prompt_dict())
        except Exception as exc:
            print(f"[agent] context error: {exc}", file=sys.stderr)
            return {}

    # -- generation --------------------------------------------------------- #
    def _client_or_build(self):
        """Return the injected Gemini client or lazily build one on first use."""
        if self._client is None:
            self._client = build_gemini_client()
        return self._client

    def _mock_line(self, ev: dict) -> str:
        """Deterministic, faithful-ish line so the loop is testable offline."""
        etype = ev.get("type", {}).get("name", "?")
        team = ev.get("team", {}).get("name", "")
        player = (ev.get("player") or {}).get("name", "").split(" ")[-1]
        tag = f"[{self.language}|{self.state.minute:02d}']"
        if etype == "Shot" and (ev.get("shot", {}).get("outcome") or {}).get("name") == "Goal":
            return f"{tag} GOAL! {player} scores for {team}! {self.state.scoreline()}."
        if etype == "Shot":
            return f"{tag} {player} ({team}) shoots — {(ev['shot']['outcome']).get('name')}."
        if etype == "Goal Keeper":
            return f"{tag} Save by {player or 'the keeper'}!"
        return f"{tag} {etype} — {team}{('/' + player) if player else ''}."

    def _generate(self, ev: dict) -> Optional[str]:
        """Generate one commentary line, using mock output or Gemini."""
        context = self.fetch_context(ev)
        user = prompts.build_event_prompt(ev, self.state.as_prompt_dict(), context)
        if self.mock:
            return self._mock_line(ev)
        try:
            from google.genai import types  # lazy import
            client = self._client_or_build()
            resp = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    temperature=0.85,
                    max_output_tokens=160,
                ),
            )
            text = (resp.text or "").strip()
        except Exception as exc:  # keep the live loop alive on any API hiccup
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                # Free-tier Gemini allows only a few requests/minute; one call per
                # commentary-worthy event blows past it. Collapse the flood into a
                # single concise hint instead of dumping the full quota JSON.
                if not getattr(self, "_rate_limit_warned", False):
                    print("[agent] Gemini rate limit hit (HTTP 429). The free tier is "
                          "very low (~5 req/min). Use --mock for the full match, demo "
                          "real Gemini on a short clip/the sample, or enable billing.",
                          file=sys.stderr)
                    self._rate_limit_warned = True
            else:
                print(f"[agent] generation error: {msg[:200]}", file=sys.stderr)
            return None
        if not text or text.upper().startswith("NO_COMMENT"):
            return None
        return text

    # -- public API --------------------------------------------------------- #
    def handle(self, ev: dict) -> Optional[str]:
        """Process one event; return a commentary line or None."""
        self._advance_clock(ev)
        will_comment = self.should_comment(ev)
        self._apply_score(ev)  # score updated before we generate the goal line
        if not will_comment:
            return None
        line = self._generate(ev)
        if line:
            self.state.last_comment_s = self.state.match_seconds()
            self.state.recent_lines.append(line)
        return line

    def run(self, events: Iterable[dict]) -> Iterator[tuple]:
        """Consume an event stream (e.g. the replayer) and yield (event, line)."""
        for ev in events:
            line = self.handle(ev)
            if line:
                yield ev, line


# --------------------------------------------------------------------------- #
# Gemini client                                                               #
# --------------------------------------------------------------------------- #
def build_gemini_client():
    """API-key client by default; Vertex AI / Agent Builder if configured."""
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("google-genai not installed — `pip install google-genai`") from exc

    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true":
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY (or run with --mock). See .env.example.")
    return genai.Client(api_key=api_key)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _load_events(match_id: Optional[int], use_sample: bool) -> list[dict]:
    """Load bundled sample events or a previously cached real match."""
    if use_sample or match_id is None:
        return json.loads((REPO / "spike" / "sample_events.json").read_text(encoding="utf-8"))
    cache = REPO / "data" / "cache" / str(match_id) / "events.json"
    if not cache.exists():
        raise SystemExit(
            f"No cached events for match {match_id}. "
            f"Run: python data/loader.py --match-id {match_id}"
        )
    return json.loads(cache.read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    """Run the Day 2 CLI demo by wiring cached/sample events into the agent."""
    parser = argparse.ArgumentParser(description="Day 2 commentary loop (replayer + agent).")
    parser.add_argument("--match-id", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help="Use bundled sample events.")
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"),
                        choices=prompts.SUPPORTED_LANGUAGE_CODES)
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Replay speed (match-sec/real-sec); 0 = no waiting.")
    parser.add_argument("--mock", action="store_true",
                        help="Offline: deterministic lines, no Gemini call.")
    args = parser.parse_args(argv)

    if not args.mock:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass

    from replayer.event_replayer import replay  # local import to avoid a hard cycle

    events = _load_events(args.match_id, args.sample)
    agent = CommentaryAgent(language=args.language, mock=args.mock)

    print(f"# MlangCast — {prompts.language_display_name(args.language)} — "
          f"{'sample' if args.match_id is None else args.match_id} "
          f"({'mock' if args.mock else agent.model})\n")
    for ev, line in agent.run(replay(events, speed=args.speed)):
        print(f"{ev.get('minute', 0):02d}:{ev.get('second', 0):02d}  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
