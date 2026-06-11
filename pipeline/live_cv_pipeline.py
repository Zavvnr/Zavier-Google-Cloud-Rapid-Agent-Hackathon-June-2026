"""
pipeline/live_cv_pipeline.py — SCAFFOLD (future: live computer-vision commentary).

MlangCast today reads STRUCTURED events (StatsBomb JSON) via the replayer. This
module sketches the ONE seam needed to go from replayed historical data to a LIVE
broadcast: swap the event SOURCE. Everything downstream — the commentary agent,
pacing/importance, MCP context retrieval, and TTS — stays byte-for-byte identical,
because they all consume the same event-dict schema.

    [ live video ] --> FrameSource --> VisionEventDetector --> LiveEventAdapter
                                                                     |
                                                  same event dicts the replayer emits
                                                                     |
                                                                     v
                                         pipeline.stream_commentary(...)  -->  text + TTS

STATUS: this is an INTERFACE scaffold to document and justify the live path — the
computer-vision pieces are stubs (raise NotImplementedError). No model is bundled.
A runnable Mock* implementation is included so the *seam* can be demonstrated
end-to-end (fake detections -> real commentary) without any CV model.

Why this is the right design: the hard part of live AI commentary is perception
(tracking + action recognition + pitch homography). By isolating that behind
`VisionEventDetector` and mapping its output to the existing event schema, the live
transition becomes a source swap rather than a rewrite of the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Contracts                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One video frame (or short clip) from a live source."""

    seconds: float                       # match clock for this frame
    image: object = None                 # raw pixels / ndarray / path — opaque here
    meta: dict = field(default_factory=dict)


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields frames in order: RTMP/HLS stream, camera, or video file."""

    def frames(self) -> Iterator[Frame]:
        ...


@dataclass
class Detection:
    """What the vision model sees in a frame — the CV output contract."""

    label: str                           # "pass" | "shot" | "goal" | "tackle" | "foul" | ...
    team: str = ""
    player: str = ""
    location: Optional[list] = None      # [x, y] on a 120x80 pitch (post-homography)
    end_location: Optional[list] = None
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)


class VisionEventDetector:
    """
    Turn frames into Detections. THE piece a future integration must supply:
    player+ball tracking, action recognition, and pitch homography (pixels -> pitch
    coordinates). Out of scope for this hackathon — stubbed on purpose.
    """

    def detect(self, frame: Frame) -> List[Detection]:
        raise NotImplementedError(
            "Plug a CV stack here (e.g. YOLO/ByteTrack tracking + an action "
            "classifier + homography). This scaffold defines the contract, not the model."
        )


_LABEL_TO_TYPE = {
    "pass": "Pass", "shot": "Shot", "goal": "Shot", "tackle": "Duel",
    "foul": "Foul Committed", "save": "Goal Keeper", "throw_in": "Pass",
}


class LiveEventAdapter:
    """
    Map Detections -> the SAME event-dict schema the StatsBomb replayer emits, so
    everything downstream is unchanged. This is the whole point: only the event
    SOURCE changes for live; the agent does not.
    """

    min_confidence: float = 0.4

    def to_events(self, detections: List[Detection], seconds: float) -> List[dict]:
        events: List[dict] = []
        minute, second = divmod(int(seconds), 60)
        for d in detections:
            if d.confidence and d.confidence < self.min_confidence:
                continue                                  # drop low-confidence noise
            ev = {
                "type": {"name": _LABEL_TO_TYPE.get(d.label, d.label.title())},
                "minute": minute, "second": second,
                "team": {"name": d.team} if d.team else {},
                "player": {"name": d.player} if d.player else {},
            }
            if d.location:
                ev["location"] = d.location
            if d.label == "pass":
                ev["pass"] = {"end_location": d.end_location} if d.end_location else {}
            elif d.label in ("shot", "goal"):
                ev["shot"] = {"outcome": {"name": "Goal" if d.label == "goal" else "Saved"}}
            events.append(ev)
        return events


# --------------------------------------------------------------------------- #
# The live source swap (drop-in for replayer.replay)                          #
# --------------------------------------------------------------------------- #
def live_event_stream(source: FrameSource, detector: VisionEventDetector,
                      adapter: Optional[LiveEventAdapter] = None) -> Iterator[dict]:
    """frames -> CV detections -> event dicts. Wire this exactly where the replayer
    is wired today; the agent consumes the output identically."""
    adapter = adapter or LiveEventAdapter()
    for frame in source.frames():
        for ev in adapter.to_events(detector.detect(frame), frame.seconds):
            yield ev


def live_commentary_stream(source: FrameSource, detector: VisionEventDetector,
                           language: str = "en", **kwargs):
    """
    The live path end-to-end: live frames -> CV -> event dicts -> the EXISTING
    commentary agent + TTS. Reuses pipeline.stream_commentary unchanged, proving the
    live transition is a source swap, not a rewrite. `kwargs` pass straight through
    (mock, tts_enabled, two_speakers, ...).
    """
    from pipeline.commentary_pipeline import stream_commentary
    return stream_commentary(live_event_stream(source, detector), language=language, **kwargs)


# --------------------------------------------------------------------------- #
# Mock CV so the SEAM is runnable without a model (demo / argument only)       #
# --------------------------------------------------------------------------- #
class MockFrameSource:
    """Emit a few frames at 5s spacing — a stand-in for a live feed."""

    def __init__(self, n: int = 6):
        self.n = n

    def frames(self) -> Iterator[Frame]:
        for i in range(self.n):
            yield Frame(seconds=float(i * 5))


class MockVisionEventDetector(VisionEventDetector):
    """Deterministic fake 'detections' so live_commentary_stream runs end-to-end
    WITHOUT a CV model — enough to demo the seam, not a real perception system."""

    _SCRIPT = [
        ("pass", "Argentina", "Rodrigo De Paul", [60, 40], [88, 30]),
        ("pass", "Argentina", "Lionel Messi", [88, 30], [104, 38]),
        ("shot", "Argentina", "Lionel Messi", [104, 38], None),
        ("pass", "France", "Kylian Mbappe", [50, 45], [80, 50]),
        ("goal", "France", "Kylian Mbappe", [108, 40], None),
        ("pass", "France", "Antoine Griezmann", [40, 30], [70, 35]),
    ]

    def __init__(self):
        self._i = 0

    def detect(self, frame: Frame) -> List[Detection]:
        if self._i >= len(self._SCRIPT):
            return []
        label, team, player, loc, end = self._SCRIPT[self._i]
        self._i += 1
        return [Detection(label=label, team=team, player=player,
                          location=loc, end_location=end, confidence=0.9)]


if __name__ == "__main__":
    # Demo the SEAM with mock CV: fake detections -> the real commentary agent.
    print("# Live-CV seam demo (mock detections -> commentary agent)\n")
    for out in live_commentary_stream(MockFrameSource(), MockVisionEventDetector(),
                                      language="en", mock=True):
        print(f"{out.event.get('minute', 0):02d}:{out.event.get('second', 0):02d}  {out.text}")
