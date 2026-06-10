"""
tts/multispeaker.py  —  SCAFFOLD (Feature 2 audio: two-voice, sequential).

Renders a list of dialogue turns (lead + analyst) to audio that PLAYS
SEQUENTIALLY — never overlapping. Two synthetic voices over each other are
unintelligible and aren't how a real goal sounds (lead's crescendo, THEN analyst).

Two paths (from the instructions):
  * Path A — Gemini-TTS NATIVE multi-speaker: one call, Speaker1/Speaker2, audio
             tags. Voice *differentiation* can be unreliable.            [stubbed]
  * Path B — two SINGLE-speaker calls (one per turn, distinct voices), played in
             order. Reliable distinct voices. Reuses tts.speak.GoogleCloudSpeaker.
  Recommendation: try A; if voices don't sound distinct, fall back to B.

Gotchas handled here: retry on the TTS 500 / text-token failure; audio TAGS (not
SSML); short chunks; strict sequential ordering.

STATUS: ordering / retry / Path-B-fallback plumbing is implemented + testable; the
real Gemini-TTS multi-speaker call is a stub (inject `multispeaker_transport` or
use mock). Path B delegates to your existing GoogleCloudSpeaker.

A turn is any object with `.speaker` and `.text` (e.g. commentary_crew.Turn).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Two DISTINCT voices per role (Path B), so the lead and analyst sound different.
# Any locale not listed falls back to the locale default voice for both roles
# (still audible, just not distinct). lead = male-ish, analyst = a contrasting voice.
ROLE_VOICES = {
    "en-US": {"lead": "en-US-Neural2-D", "analyst": "en-US-Neural2-A"},
    "en-GB": {"lead": "en-GB-Neural2-B", "analyst": "en-GB-Neural2-A"},
    "es-ES": {"lead": "es-ES-Neural2-B", "analyst": "es-ES-Neural2-C"},
    "es-US": {"lead": "es-US-Neural2-B", "analyst": "es-US-Neural2-A"},
    "pt-BR": {"lead": "pt-BR-Neural2-B", "analyst": "pt-BR-Neural2-A"},
    "fr-FR": {"lead": "fr-FR-Neural2-B", "analyst": "fr-FR-Neural2-A"},
    "de-DE": {"lead": "de-DE-Neural2-B", "analyst": "de-DE-Neural2-C"},
    "it-IT": {"lead": "it-IT-Neural2-C", "analyst": "it-IT-Neural2-A"},
    "nl-NL": {"lead": "nl-NL-Wavenet-B", "analyst": "nl-NL-Wavenet-A"},
    "ru-RU": {"lead": "ru-RU-Wavenet-D", "analyst": "ru-RU-Wavenet-C"},
    "tr-TR": {"lead": "tr-TR-Wavenet-B", "analyst": "tr-TR-Wavenet-A"},
    "ar-XA": {"lead": "ar-XA-Wavenet-B", "analyst": "ar-XA-Wavenet-A"},
    "hi-IN": {"lead": "hi-IN-Neural2-B", "analyst": "hi-IN-Neural2-A"},
    "ja-JP": {"lead": "ja-JP-Neural2-C", "analyst": "ja-JP-Neural2-B"},
    "ko-KR": {"lead": "ko-KR-Neural2-C", "analyst": "ko-KR-Neural2-A"},
    "cmn-CN": {"lead": "cmn-CN-Wavenet-B", "analyst": "cmn-CN-Wavenet-A"},
    "vi-VN": {"lead": "vi-VN-Wavenet-D", "analyst": "vi-VN-Wavenet-A"},
    "id-ID": {"lead": "id-ID-Wavenet-B", "analyst": "id-ID-Wavenet-A"},
}


@dataclass
class TurnAudio:
    """One spoken turn's audio (or why it was skipped)."""

    speaker: str
    text: str
    audio_bytes: Optional[bytes] = None
    audio_path: Optional[Path] = None
    skipped_reason: str = ""

    def has_audio(self) -> bool:
        return bool(self.audio_bytes)


@dataclass
class DialogueAudio:
    """Ordered per-turn audio to play sequentially (no mixing)."""

    segments: List[TurnAudio] = field(default_factory=list)

    def has_audio(self) -> bool:
        return any(s.has_audio() for s in self.segments)


def _retry(fn, attempts: int = 3, base_delay: float = 0.6):
    """Retry helper — the Gemini-TTS model occasionally 500s / returns text tokens."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — intentional broad retry
            last_exc = exc
            time.sleep(base_delay * (i + 1))
    raise last_exc


@dataclass
class MultiSpeakerSpeaker:
    """
    Synthesize a two-speaker dialogue to ordered, non-overlapping audio.

    Seams for testing/wiring (no creds needed to import):
      * mock=True                  -> returns placeholder audio per turn.
      * multispeaker_transport     -> Path A: callable(turns, language) -> [bytes].
      * single_speaker             -> Path B: a GoogleCloudSpeaker-like object.
    """

    language: str = "en-US"
    path: str = "A"                 # "A" multi-speaker, "B" two single-speaker calls
    mock: bool = False
    multispeaker_transport: Optional[Callable[[list, str], List[bytes]]] = None
    single_speaker: object = None

    def synthesize_dialogue(self, turns: list) -> DialogueAudio:
        """Render `turns` to sequential audio, with Path A -> Path B fallback."""
        if self.mock:
            return DialogueAudio([TurnAudio(t.speaker, t.text, b"MOCK_AUDIO") for t in turns])
        if self.path == "A":
            try:
                return self._path_a(turns)
            except Exception:
                pass  # voices not distinct / transport unavailable -> fall back to B
        return self._path_b(turns)

    # -- Path A: native multi-speaker (one call) ---------------------------- #
    def _path_a(self, turns: list) -> DialogueAudio:
        if self.multispeaker_transport is None:
            # TODO: implement the Gemini-TTS multi-speaker call here
            #   (gemini-3.1-flash-tts-preview / 2.5 TTS; Speaker1->lead, Speaker2->analyst;
            #    audio tags, not SSML). See references in instructions.md.
            raise RuntimeError("Path A transport not wired")
        blobs = _retry(lambda: self.multispeaker_transport(turns, self.language))
        if len(blobs) != len(turns):
            raise RuntimeError("Path A returned the wrong number of audio segments")
        return DialogueAudio([
            TurnAudio(t.speaker, t.text, audio_bytes=b) for t, b in zip(turns, blobs)
        ])

    # -- Path B: per-turn single-speaker calls (reliable distinct voices) --- #
    def _path_b(self, turns: list) -> DialogueAudio:
        voices = ROLE_VOICES.get(self.language, {})
        segments: List[TurnAudio] = []
        for t in turns:
            voice_name = voices.get(t.speaker)
            try:
                result = _retry(lambda txt=t.text, vn=voice_name:
                                _synth_one(txt, self.language, vn, self.single_speaker))
                segments.append(TurnAudio(
                    t.speaker,
                    t.text,
                    audio_bytes=getattr(result, "audio_bytes", None),
                    audio_path=getattr(result, "audio_path", None),
                    skipped_reason=getattr(result, "skipped_reason", "") or "",
                ))
            except Exception as exc:  # noqa: BLE001
                segments.append(TurnAudio(
                    t.speaker,
                    t.text,
                    skipped_reason=f"{type(exc).__name__}: {exc}",
                ))
        return DialogueAudio(segments)


def _synth_one(text: str, language: str, voice_name: Optional[str], injected=None):
    """Synthesize one turn via the injected speaker, or a fresh GoogleCloudSpeaker."""
    if injected is not None:
        return injected.synthesize(text, language=language)
    from tts.speak import GoogleCloudSpeaker  # lazy: your existing single-speaker TTS
    return GoogleCloudSpeaker(voice_name=voice_name).synthesize(text, language=language)


def build_multispeaker_speaker(language: str = "en-US", path: str = "A", **kwargs) -> MultiSpeakerSpeaker:
    """Factory mirroring tts.speak.build_speaker for consistency."""
    return MultiSpeakerSpeaker(language=language, path=path, **kwargs)
