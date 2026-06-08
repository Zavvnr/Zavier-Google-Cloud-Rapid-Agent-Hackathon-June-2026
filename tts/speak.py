"""
Text-to-speech for the commentary pipeline.

  * NoOpSpeaker        — default; returns the text with no audio
                         (no credentials, runs anywhere).
  * GoogleCloudSpeaker — real synthesis via the Google Cloud Text-to-Speech REST
                         API authenticated with GOOGLE_API_KEY. Saves an mp3 and
                         returns its bytes/path.

Why REST + API key (not the google-cloud-texttospeech client library)? The library
authenticates with Application Default Credentials (a service account), but the
only credential available here is GOOGLE_API_KEY — and Cloud TTS accepts an API
key on the REST endpoint (`...:synthesize?key=...`). This keeps setup to a single
env var.

FAIL-SAFE: no API key, a network error, or a bad voice degrades to a text-only
SpeechResult (skipped_reason set) instead of raising — the live commentary loop
must never break because audio failed.

CLI:
    python -m tts.speak --text "Goal for Argentina!" --language es-ES
    python -m tts.speak --text "Gol!" --language es-ES --out tts/out
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO / "tts" / "out"
TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Nice default voices for the demo languages. Any locale not listed falls back to
# "languageCode only", letting Cloud TTS pick a default voice for that locale.
DEFAULT_VOICES = {
    "en-US": "en-US-Neural2-D",
    "en-GB": "en-GB-Neural2-B",
    "es-ES": "es-ES-Neural2-B",
    "es-US": "es-US-Neural2-B",
    "fr-FR": "fr-FR-Neural2-B",
    "pt-BR": "pt-BR-Neural2-B",
    "de-DE": "de-DE-Neural2-B",
    "id-ID": "id-ID-Standard-B",
}


@dataclass
class SpeechResult:
    """Container for text plus optional audio produced by a TTS provider."""

    text: str
    language: str
    provider: str
    audio_path: Optional[Path] = None
    audio_bytes: Optional[bytes] = None
    mime_type: str = "audio/mpeg"
    skipped_reason: str = ""

    def has_audio(self) -> bool:
        """Return True when a real TTS provider produced audio output."""
        return bool(self.audio_path or self.audio_bytes)


@dataclass
class NoOpSpeaker:
    """Offline speaker used by default; returns text and never writes audio."""

    provider: str = "noop"

    def synthesize(
        self,
        text: str,
        language: str = "en-US",
        output_dir: Optional[Path] = None,
    ) -> SpeechResult:
        """Return the text unchanged and mark audio synthesis as skipped."""
        return SpeechResult(
            text=text,
            language=language,
            provider=self.provider,
            skipped_reason="TTS disabled (NoOpSpeaker).",
        )


@dataclass
class GoogleCloudSpeaker:
    """
    Real Google Cloud Text-to-Speech synthesis over the REST API + GOOGLE_API_KEY.

    `transport` is the seam for tests: a callable (url, payload, api_key) -> dict
    that mimics the REST response {"audioContent": "<base64>"}. Left None in
    production, where an internal requests-based POST is used.
    """

    voice_name: Optional[str] = None                 # force a specific voice
    language_voice_map: dict = field(default_factory=lambda: dict(DEFAULT_VOICES))
    audio_encoding: str = "MP3"
    api_key: Optional[str] = None
    endpoint: str = TTS_ENDPOINT
    provider: str = "google-cloud-tts"
    transport: Optional[Callable[[str, dict, str], dict]] = None

    def __post_init__(self) -> None:
        """Pull the API key from the environment (names only; no .env parsing)."""
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")

    def _voice_options(self, language: str) -> list[dict]:
        """Voice payloads to try, in order (named voice first, then locale-only)."""
        if self.voice_name:
            primary = {"languageCode": language, "name": self.voice_name}
        else:
            name = self.language_voice_map.get(language)
            primary = {"languageCode": language}
            if name:
                primary["name"] = name
        options = [primary]
        if "name" in primary:
            options.append({"languageCode": language})  # fallback: let Cloud choose
        return options

    def _post(self, payload: dict) -> dict:
        """Default REST transport (overridden in tests via `transport`)."""
        if self.transport is not None:
            return self.transport(self.endpoint, payload, self.api_key or "")
        import requests  # imported lazily so importing this module needs no requests
        resp = requests.post(
            self.endpoint, params={"key": self.api_key}, json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def synthesize(
        self,
        text: str,
        language: str = "en-US",
        output_dir: Optional[Path] = None,
    ) -> SpeechResult:
        """Synthesize `text` to an mp3; fall back to text-only on any failure."""
        result = SpeechResult(text=text, language=language, provider=self.provider)
        if not text or not text.strip():
            result.skipped_reason = "empty text"
            return result
        if self.transport is None and not self.api_key:
            result.skipped_reason = "GOOGLE_API_KEY not set; returning text only."
            return result

        last_error = ""
        for voice in self._voice_options(language):
            payload = {
                "input": {"text": text},
                "voice": voice,
                "audioConfig": {"audioEncoding": self.audio_encoding},
            }
            try:
                data = self._post(payload)
                audio_b64 = data.get("audioContent")
                if not audio_b64:
                    last_error = "no audioContent in response"
                    continue
                audio = base64.b64decode(audio_b64)
            except Exception as exc:  # network/auth/bad-voice -> try next, then skip
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            out_dir = Path(output_dir) if output_dir else DEFAULT_OUT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(f"{language}|{text}".encode("utf-8")).hexdigest()[:12]
            path = out_dir / f"{language}-{digest}.mp3"
            path.write_bytes(audio)
            result.audio_path = path
            result.audio_bytes = audio
            return result

        result.skipped_reason = f"TTS failed: {last_error}"
        return result


def build_speaker(enabled: bool = False, provider: str = "noop", **kwargs) -> object:
    """
    Create a TTS provider for the commentary pipeline.

    Default is the offline NoOpSpeaker. Pass enabled=True and provider="google"
    to use real Google Cloud TTS. (Keeping noop the default is what lets the
    pipeline/tests stay credential-free.)
    """
    if enabled and provider == "google":
        return GoogleCloudSpeaker(**kwargs)
    return NoOpSpeaker()


def main(argv: Optional[list[str]] = None) -> int:
    """Synthesize one line from the command line (handy for checking a voice)."""
    parser = argparse.ArgumentParser(description="Synthesize one line via Google Cloud TTS.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en-US"))
    parser.add_argument("--out", type=Path, default=None, help="Output directory for the mp3.")
    parser.add_argument("--voice", default=None, help="Override the Cloud TTS voice name.")
    args = parser.parse_args(argv)

    speaker = GoogleCloudSpeaker(voice_name=args.voice)
    result = speaker.synthesize(args.text, language=args.language, output_dir=args.out)
    if result.has_audio():
        print(f"OK  -> {result.audio_path}  ({len(result.audio_bytes or b'')} bytes)")
    else:
        print(f"SKIPPED ({result.skipped_reason}) — text: {result.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
