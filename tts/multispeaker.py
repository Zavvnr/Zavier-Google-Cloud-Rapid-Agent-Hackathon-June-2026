"""
tts/multispeaker.py  —  SCAFFOLD (Feature 2 audio: two-voice, sequential).

Renders a list of dialogue turns (lead + analyst) to audio that PLAYS
SEQUENTIALLY — never overlapping. Two synthetic voices over each other are
unintelligible and aren't how a real goal sounds (lead's crescendo, THEN analyst).

Two paths (from the instructions):
  * Path A — Gemini-TTS, PROMPTABLE style/emotion (the "sports commentator" feel,
             since Cloud TTS has no dedicated sports voice). Per-turn, distinct
             voices + a style prompt; WAV output written to the TTS out dir.
  * Path B — two SINGLE-speaker calls (one per turn, distinct Chirp 3: HD voices),
             played in order. Reliable. Reuses tts.speak.GoogleCloudSpeaker.
  Path A auto-falls back to Path B on ANY error (model unavailable, no key, etc.).

Gotchas handled here: retry on the TTS 500 / text-token failure; audio TAGS folded
into the style prompt; PCM->WAV wrapping; strict sequential ordering.

STATUS: both paths implemented. Path A calls Gemini-TTS (needs google-genai + key +
GEMINI_TTS_MODEL); inject `multispeaker_transport` to test it offline. Path B
delegates to your existing GoogleCloudSpeaker (Chirp 3: HD).

A turn is any object with `.speaker` and `.text` (e.g. commentary_crew.Turn).
"""
from __future__ import annotations

import hashlib
import io
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Gemini-TTS model (promptable style/emotion, single- or multi-speaker). Override
# via env if Google renames it; Path A auto-falls back to Path B if it fails.
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")

# Path A role -> Gemini-TTS prebuilt voice + a natural-language STYLE direction.
# The style is what turns a generic voice into a "sports commentator" — Cloud TTS
# has no dedicated sports voice, so you prompt one.
GEMINI_ROLES = {
    "lead":    {"voice": "Charon",
                "style": "an energetic football play-by-play commentator — fast, vivid, building excitement"},
    "analyst": {"voice": "Aoede",
                "style": "a warm, insightful co-commentator reacting calmly just after the lead"},
}

# Two DISTINCT voices per role (Path B), so the lead and analyst sound different.
# Any locale not listed falls back to the locale default voice for both roles
# (still audible, just not distinct). lead = male-ish, analyst = a contrasting voice.
ROLE_VOICES = {
    "en-US": {"lead": "en-US-Chirp3-HD-Charon", "analyst": "en-US-Chirp3-HD-Aoede"},
    "en-GB": {"lead": "en-GB-Neural2-B", "analyst": "en-GB-Neural2-A"},
    "es-ES": {"lead": "es-ES-Chirp3-HD-Charon", "analyst": "es-ES-Chirp3-HD-Aoede"},
    "es-US": {"lead": "es-US-Neural2-B", "analyst": "es-US-Neural2-A"},
    "pt-BR": {"lead": "pt-BR-Chirp3-HD-Charon", "analyst": "pt-BR-Chirp3-HD-Aoede"},
    "fr-FR": {"lead": "fr-FR-Chirp3-HD-Charon", "analyst": "fr-FR-Chirp3-HD-Aoede"},
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
    "id-ID": {"lead": "id-ID-Chirp3-HD-Charon", "analyst": "id-ID-Chirp3-HD-Aoede"},
    "ms-MY": {"lead": "ms-MY-Chirp3-HD-Charon", "analyst": "ms-MY-Chirp3-HD-Aoede"},
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

    def synthesize_dialogue(self, turns: list, speaking_rate: Optional[float] = None) -> DialogueAudio:
        """Render `turns` to sequential audio, with Path A -> Path B fallback.

        `speaking_rate` (forwarded to per-turn TTS in Path B) lets the agent push
        the tempo up for intense moments. Path A (stub) ignores it for now.
        """
        if self.mock:
            return DialogueAudio([TurnAudio(t.speaker, t.text, b"MOCK_AUDIO") for t in turns])
        if self.path == "A":
            try:
                return self._path_a(turns)
            except Exception:
                pass  # voices not distinct / transport unavailable -> fall back to B
        return self._path_b(turns, speaking_rate=speaking_rate)

    # -- Path A: Gemini-TTS, promptable "sports" style ---------------------- #
    def _path_a(self, turns: list) -> DialogueAudio:
        """Synthesize each turn with Gemini-TTS (style-prompted), writing files so
        the web can serve them. Raises on any failure so the caller falls back to B.

        Uses the injected `multispeaker_transport` when provided (tests), otherwise
        the real per-turn Gemini-TTS transport.
        """
        transport = self.multispeaker_transport or gemini_multispeaker_transport
        blobs = _retry(lambda: transport(turns, self.language))
        if len(blobs) != len(turns):
            raise RuntimeError("Path A returned the wrong number of audio segments")
        segments: List[TurnAudio] = []
        for t, blob in zip(turns, blobs):
            path = _write_audio(blob, self.language, t.text) if blob else None
            segments.append(TurnAudio(t.speaker, t.text, audio_bytes=blob, audio_path=path))
        if not any(s.has_audio() for s in segments):
            raise RuntimeError("Path A produced no audio")
        return DialogueAudio(segments)

    # -- Path B: per-turn single-speaker calls (reliable distinct voices) --- #
    def _path_b(self, turns: list, speaking_rate: Optional[float] = None) -> DialogueAudio:
        voices = ROLE_VOICES.get(self.language, {})
        segments: List[TurnAudio] = []
        for t in turns:
            voice_name = voices.get(t.speaker)
            try:
                result = _retry(lambda txt=t.text, vn=voice_name:
                                _synth_one(txt, self.language, vn, self.single_speaker,
                                           speaking_rate=speaking_rate))
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


def _synth_one(text: str, language: str, voice_name: Optional[str], injected=None,
               speaking_rate: Optional[float] = None):
    """Synthesize one turn via the injected speaker, or a fresh GoogleCloudSpeaker."""
    if injected is not None:
        return injected.synthesize(text, language=language)
    from tts.speak import GoogleCloudSpeaker  # lazy: your existing single-speaker TTS
    return GoogleCloudSpeaker(voice_name=voice_name).synthesize(
        text, language=language, speaking_rate=speaking_rate)


# --------------------------------------------------------------------------- #
# Path A transport — Gemini-TTS (promptable style + emotion).                  #
# --------------------------------------------------------------------------- #
def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM (Gemini-TTS output) in a WAV container (stdlib)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _gemini_tts_turn(text: str, voice: str, style: str, language: str) -> bytes:
    """Synthesize ONE turn with Gemini-TTS; returns WAV bytes. Needs google-genai + key."""
    from google import genai            # lazy: only imported when Path A actually runs
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = f"Read this as {style}. Language: {language}.\n{text}"
    resp = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    pcm = resp.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav(pcm)


def gemini_multispeaker_transport(turns: list, language: str) -> List[bytes]:
    """Default Path A transport: per-turn Gemini-TTS with a sports style, one WAV
    blob per turn (so it slots into the existing sequential DialogueAudio). The
    script's [excited]-style audio tags are folded into the delivery direction."""
    blobs: List[bytes] = []
    for t in turns:
        role = GEMINI_ROLES.get(getattr(t, "speaker", "lead"), GEMINI_ROLES["lead"])
        tags = " ".join(getattr(t, "audio_tags", []) or [])
        style = role["style"] + (f"; delivery: {tags}" if tags else "")
        blobs.append(_gemini_tts_turn(t.text, role["voice"], style, language))
    return blobs


def _write_audio(audio: bytes, language: str, text: str, ext: str = "wav") -> Path:
    """Persist a turn's audio under the TTS out dir the web serves from."""
    from tts.speak import DEFAULT_OUT_DIR
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{language}|gemini|{text}".encode("utf-8")).hexdigest()[:12]
    path = DEFAULT_OUT_DIR / f"{language}-{digest}.{ext}"
    path.write_bytes(audio)
    return path


def build_multispeaker_speaker(language: str = "en-US", path: str = "A", **kwargs) -> MultiSpeakerSpeaker:
    """Factory mirroring tts.speak.build_speaker for consistency."""
    return MultiSpeakerSpeaker(language=language, path=path, **kwargs)
