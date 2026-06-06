"""
replayer/event_replayer.py  —  Day 2 deliverable.

Replays a StatsBomb match event stream in accelerated real time, so the
commentary agent receives events with realistic pacing (a 3-second build-up takes
3 match-seconds) instead of all at once.

Design notes:
  * Transport only. The replayer never decides what's interesting (that's the
    agent) and never mutates events.
  * Pacing comes from each event's `timestamp` ("HH:MM:SS.mmm") within its period.
  * The half-time gap is skipped: when `period` changes we don't sleep across it.
  * `sleep` is injectable so tests can run instantly (pass a no-op).

CLI:
    python -m replayer.event_replayer --match-id 3869685 --speed 60
    python -m replayer.event_replayer --sample --speed 0      # bundled sample, no waiting
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "spike" / "sample_events.json"


def parse_timestamp(ts: str) -> float:
    """'HH:MM:SS.mmm' -> seconds (float) elapsed within the current period."""
    hours, minutes, seconds = ts.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _order_key(ev: dict) -> tuple:
    """Return a stable ordering key that prefers StatsBomb's canonical index."""
    # StatsBomb `index` is the canonical within-match ordering; fall back to the
    # clock if it's ever missing.
    return (ev.get("index", 0), ev.get("period", 0), ev.get("minute", 0), ev.get("second", 0))


def replay(
    events: Iterable[dict],
    speed: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    on_event: Optional[Callable[[dict], None]] = None,
) -> Iterator[dict]:
    """
    Yield events one at a time, waiting between them in proportion to the gap in
    match time. `speed` is match-seconds per wall-clock second (60 => a minute of
    football per real second). `speed <= 0` disables waiting (as fast as possible).

    Yields each event after its wait, so consumers can simply iterate:
        for event in replay(events, speed=60):
            agent.handle(event)
    """
    ordered = sorted(events, key=_order_key)
    prev_period: Optional[int] = None
    prev_ts: Optional[float] = None

    for ev in ordered:
        period = ev.get("period", 1)
        ts = parse_timestamp(ev.get("timestamp", "00:00:00.000"))

        if speed > 0 and prev_ts is not None and period == prev_period:
            wait = max(0.0, ts - prev_ts) / speed
            if wait > 0:
                sleep(wait)

        prev_period, prev_ts = period, ts
        if on_event is not None:
            on_event(ev)
        yield ev


def summarize_event(ev: dict) -> str:
    """Compact one-line view for CLI/logging."""
    minute = ev.get("minute", 0)
    second = ev.get("second", 0)
    etype = ev.get("type", {}).get("name", "?")
    team = ev.get("team", {}).get("name", "")
    player = (ev.get("player") or {}).get("name", "")
    tail = f" — {team}" + (f" / {player}" if player else "") if team else ""
    return f"P{ev.get('period', 1)} {minute:02d}:{second:02d}  {etype}{tail}"


def _load_events(match_id: Optional[int], use_sample: bool) -> list[dict]:
    """Load bundled sample events or cached match events for the CLI."""
    if use_sample or match_id is None:
        return json.loads(SAMPLE.read_text(encoding="utf-8"))
    cache = REPO / "data" / "cache" / str(match_id) / "events.json"
    if not cache.exists():
        raise SystemExit(
            f"No cached events for match {match_id}. "
            f"Run: python data/loader.py --match-id {match_id}"
        )
    return json.loads(cache.read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    """Run the replayer CLI and print each replayed event summary."""
    parser = argparse.ArgumentParser(description="Replay a match event stream in accelerated time.")
    parser.add_argument("--match-id", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help="Use bundled sample events.")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="Match-seconds per real second (0 = no waiting).")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N events.")
    args = parser.parse_args(argv)

    events = _load_events(args.match_id, args.sample)
    for i, ev in enumerate(replay(events, speed=args.speed)):
        print(summarize_event(ev))
        if args.limit is not None and i + 1 >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
