"""
spike/go_no_go.py

The whole point of this script is to answer ONE question before building anything:
    "If we hand Gemini ~15-20 real match events, is the commentary actually good?"

So this script is deliberately throwaway — no replayer, no agent, no MCP, no TTS.
Events in -> one Gemini call -> commentary out -> a human eyeballs it.

Run:
    python spike/go_no_go.py                         # bundled illustrative sample
    python spike/go_no_go.py --language es           # Spanish
    python spike/go_no_go.py --match-id 3869685 --start 1500 --count 18
                                                     # a real cached match slice
                                                     # (run data/loader.py first)
    python spike/go_no_go.py --mock                  # offline: print the prompt, no API call

NOTE: spike/sample_events.json is an ILLUSTRATIVE, StatsBomb-shaped sample so this
runs with no download. Player/team ids are not canonical. For the real go/no-go,
point it at a downloaded match with --match-id.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SAMPLE = HERE / "sample_events.json"

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "id": "Indonesian",
    "fr": "French", "pt": "Portuguese", "de": "German",
}

# Event types that carry no narrative weight on their own — drop them so the
# ~18-event window the model sees is dense with meaningful play.
SKIP_TYPES = {"Ball Receipt*", "Pressure", "Carry", "Half Start", "Half End"}


# --------------------------------------------------------------------------- #
# Load events                                                                 #
# --------------------------------------------------------------------------- #
def load_events(match_id: Optional[int]) -> list[dict]:
    """Load the bundled sample or a cached real match for the spike."""
    if match_id is None:
        return json.loads(SAMPLE.read_text(encoding="utf-8"))
    # Reuse the loader's cache layout.
    cache = REPO / "data" / "cache" / str(match_id) / "events.json"
    if not cache.exists():
        raise SystemExit(
            f"No cached events for match {match_id}.\n"
            f"Download it first:  python data/loader.py --match-id {match_id}"
        )
    return json.loads(cache.read_text(encoding="utf-8"))


def select_window(events: list[dict], start: int, count: int, dense: bool) -> list[dict]:
    """Take a contiguous slice, optionally dropping low-signal event types."""
    window = events[start:start + count]
    if dense:
        window = [e for e in window if e.get("type", {}).get("name") not in SKIP_TYPES]
    return window


# --------------------------------------------------------------------------- #
# Format events for the model                                                 #
# --------------------------------------------------------------------------- #
def describe_event(ev: dict) -> str:
    """One compact, faithful line per event — only facts present in the data."""
    minute = ev.get("minute", 0)
    second = ev.get("second", 0)
    etype = ev.get("type", {}).get("name", "?")
    team = ev.get("team", {}).get("name", "?")
    player = (ev.get("player") or {}).get("name", "")
    clock = f"{minute:02d}:{second:02d}"
    head = f"[{clock}] {etype} — {team}"
    if player:
        head += f" / {player}"

    detail = ""
    if etype == "Pass" and "pass" in ev:
        p = ev["pass"]
        recipient = (p.get("recipient") or {}).get("name", "")
        outcome = (p.get("outcome") or {}).get("name", "Complete")
        tech = (p.get("technique") or {}).get("name", "")
        bits = [b for b in (f"to {recipient}" if recipient else "", outcome, tech) if b]
        detail = " (" + ", ".join(bits) + ")" if bits else ""
    elif etype == "Shot" and "shot" in ev:
        s = ev["shot"]
        outcome = (s.get("outcome") or {}).get("name", "?")
        body = (s.get("body_part") or {}).get("name", "")
        xg = s.get("statsbomb_xg")
        bits = [outcome]
        if body:
            bits.append(body)
        if xg is not None:
            bits.append(f"xG {xg:.2f}")
        detail = " (" + ", ".join(bits) + ")"
    elif etype == "Dribble" and "dribble" in ev:
        detail = f" ({(ev['dribble'].get('outcome') or {}).get('name', '?')})"
    elif etype == "Goal Keeper" and "goalkeeper" in ev:
        detail = f" ({(ev['goalkeeper'].get('type') or {}).get('name', '?')})"

    return head + detail


def build_prompt(events: list[dict], language: str) -> str:
    lang_name = LANGUAGE_NAMES.get(language, language)
    feed = "\n".join(describe_event(e) for e in events)
    return f"""You are a live football commentator. Below is a short, ordered window of
real match events (newest last), already filtered to the meaningful ones.

Produce natural running commentary in {lang_name}. Rules:
- Comment ONLY on what the events state. Do NOT invent goals, names, scores,
  cards, or details that are not in the data.
- Don't narrate every line — group the build-up and land the big moments.
- Match the energy to the play: calm in midfield, loud for the shot and goal.
- Write the commentary in {lang_name} only.

EVENTS:
{feed}

Commentary ({lang_name}):"""


# --------------------------------------------------------------------------- #
# Gemini call (standalone — Has no shared client yet)                   #
# --------------------------------------------------------------------------- #
def call_gemini(prompt: str) -> str:
    """Send the assembled prompt to Gemini and return the text response."""
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("google-genai not installed — `pip install google-genai`") from exc

    model = os.getenv("GEMINI_MODEL", "gemini-3-pro")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if use_vertex:
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY (or use --mock). See .env.example.")
        client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


GO_NO_GO_CHECKLIST = """
------------------------------------------------------------------
GO / NO-GO — judge the output above:
  [ ] Faithful?    No invented goals, names, cards, or scores.
  [ ] Right language?  Entirely in the requested language.
  [ ] Paced?       Build-up grouped, big moments emphasised.
  [ ] Listenable?  Reads like a commentator, not a data dump.
If yes -> GO: proceed to replayer + agent.
If no  -> NO-GO: iterate on the prompt before building the pipeline.
------------------------------------------------------------------"""


def main(argv: Optional[list[str]] = None) -> int:
    """Run the go/no-go CLI from event selection through commentary output."""
    parser = argparse.ArgumentParser(description="go/no-go commentary spike.")
    parser.add_argument("--match-id", type=int, default=None,
                        help="Use a cached match instead of the bundled sample.")
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"),
                        choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--start", type=int, default=0, help="Window start index.")
    parser.add_argument("--count", type=int, default=18, help="Events to include (~15-20).")
    parser.add_argument("--all-types", action="store_true",
                        help="Keep low-signal events (passes' receipts, pressure, carries).")
    parser.add_argument("--mock", action="store_true",
                        help="Print the assembled prompt without calling Gemini (offline).")
    args = parser.parse_args(argv)

    # Try to load local API settings only when a real Gemini call is requested.
    if not args.mock:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass

    events = select_window(
        load_events(args.match_id), args.start, args.count, dense=not args.all_types
    )
    if not events:
        raise SystemExit("No events in the selected window — adjust --start/--count.")

    prompt = build_prompt(events, args.language)

    print(f"# {len(events)} events -> {LANGUAGE_NAMES[args.language]}  "
          f"(match={'sample' if args.match_id is None else args.match_id})\n")

    if args.mock:
        print("----- PROMPT (mock, no API call) -----")
        print(prompt)
        return 0

    print("----- GEMINI COMMENTARY -----")
    print(call_gemini(prompt))
    print(GO_NO_GO_CHECKLIST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
