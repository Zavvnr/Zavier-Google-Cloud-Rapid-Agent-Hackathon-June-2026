"""
Pipeline glue for Day 2 text, Day 3 context, and Day 4 speech.

This keeps the CLI/demo path small: events enter from the replayer, the agent
produces text, and a speaker can optionally turn that text into audio.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from agent.commentary_agent import CommentaryAgent
from agent.mcp_client import build_context_client
from agent import prompts
from replayer.event_replayer import replay
from tts.speak import SpeechResult, build_speaker

REPO = Path(__file__).resolve().parent.parent


@dataclass
class CommentaryOutput:
    """One emitted commentary item with its source event and optional speech."""

    event: dict
    text: str
    speech: SpeechResult

    def as_dict(self) -> dict:
        """Serialize one output for a web UI or JSON-lines stream."""
        return {
            "minute": self.event.get("minute", 0),
            "second": self.event.get("second", 0),
            "event_type": (self.event.get("type") or {}).get("name", ""),
            "text": self.text,
            "language": self.speech.language,
            "audio_ready": self.speech.has_audio(),
            "audio_path": str(self.speech.audio_path) if self.speech.audio_path else "",
            "tts_provider": self.speech.provider,
        }


def _env_float(name: str, default: float) -> float:
    """Read a float environment setting with a safe fallback."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def stream_commentary(
    events: Iterable[dict],
    language: str = "en",
    speed: float = 0.0,
    mock: bool = False,
    context_enabled: bool = False,
    tts_enabled: bool = False,
    tts_provider: str = "noop",
    agent: Optional[CommentaryAgent] = None,
    speaker=None,
) -> Iterator[CommentaryOutput]:
    """
    Stream replayed events through the agent and optional Day 3 context + Day 4 TTS.

    `tts_provider` defaults to "noop" (text only). Pass "google" with
    tts_enabled=True for real Google Cloud TTS audio.
    """
    context_client = build_context_client(enabled=context_enabled)
    active_agent = agent or CommentaryAgent(
        language=language,
        mock=mock,
        context_client=context_client,
    )
    active_speaker = speaker or build_speaker(enabled=tts_enabled, provider=tts_provider)

    for event in replay(events, speed=speed):
        line = active_agent.handle(event)
        if not line:
            continue
        speech = active_speaker.synthesize(line, language=active_agent.language)
        yield CommentaryOutput(event=event, text=line, speech=speech)


def _load_events(match_id: Optional[int], use_sample: bool) -> list[dict]:
    """Load sample events or cached match events for the pipeline CLI."""
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
    """Run the end-to-end pipeline (replay -> context -> agent -> TTS) as JSON lines."""
    parser = argparse.ArgumentParser(description="Replay -> context -> commentary -> TTS pipeline.")
    parser.add_argument("--match-id", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"),
                        choices=prompts.SUPPORTED_LANGUAGE_CODES)
    parser.add_argument("--speed", type=float, default=_env_float("REPLAY_SPEED", 0.0))
    parser.add_argument("--mock", action="store_true", help="Offline deterministic commentary.")
    parser.add_argument("--context", action="store_true", help="Enable Day 3 MongoDB context.")
    parser.add_argument("--tts", action="store_true", help="Enable Day 4 text-to-speech.")
    parser.add_argument("--tts-provider", default="noop", choices=["noop", "google"],
                        help="TTS backend when --tts is set (google = Google Cloud TTS).")
    args = parser.parse_args(argv)

    # Load .env so MONGODB_URI / GOOGLE_API_KEY are available for real runs.
    # (Best-effort; --mock needs nothing. The app reads its own config here.)
    if not args.mock:
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO / ".env")
        except ImportError:
            pass

    events = _load_events(args.match_id, args.sample)
    for item in stream_commentary(
        events,
        language=args.language,
        speed=args.speed,
        mock=args.mock,
        context_enabled=args.context,
        tts_enabled=args.tts,
        tts_provider=args.tts_provider,
    ):
        print(json.dumps(item.as_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
