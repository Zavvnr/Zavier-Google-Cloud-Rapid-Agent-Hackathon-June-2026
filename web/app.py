"""
Day 4: minimal web UI for MlangCast.

Pick a language + match, then watch commentary stream in near-real-time with
optional synced audio. This module is thin glue over pipeline.stream_commentary;
all the real work (replay -> context -> agent -> TTS) lives in the modules it calls.

Run locally:
    pip install flask
    python -m web.app                # serves http://localhost:8080

Endpoints:
    GET /                  -> single-page UI (web/static/index.html)
    GET /api/languages     -> [{code, name}] for the language picker
    GET /api/matches       -> [{id, label}] cached matches (plus "sample")
    GET /api/stream        -> Server-Sent Events: one commentary line per event
    GET /api/audio/<file>  -> serves an mp3 produced by the TTS step
    GET /api/tts           -> synthesize one line on demand (audio/mpeg)

Notes:
    * .env is loaded only in main() (local runs). On Cloud Run, real env vars are
      injected by the platform, so create_app() never needs to read a file.
    * Streaming uses the agent in whatever mode the query asks for: ?mock=1 gives a
      fast, offline, deterministic demo; mock=0 calls Gemini for real commentary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from agent import prompts
from pipeline.commentary_pipeline import stream_commentary
from tts.speak import DEFAULT_OUT_DIR, GoogleCloudSpeaker

REPO = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE = REPO / "spike" / "sample_events.json"
CACHE_DIR = REPO / "data" / "cache"


def _load_events(match: str) -> list[dict]:
    """Resolve a match id (or 'sample') to its event list."""
    if not match or match == "sample":
        return json.loads(SAMPLE.read_text(encoding="utf-8"))
    cache = CACHE_DIR / str(match) / "events.json"
    if not cache.exists():
        raise FileNotFoundError(f"Match {match} is not cached. Run data/loader.py first.")
    return json.loads(cache.read_text(encoding="utf-8"))


def _match_label(match_id: str) -> str:
    """Build a friendly match label from cached meta.json when available."""
    meta_path = CACHE_DIR / match_id / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            home = (meta.get("home_team") or {}).get("home_team_name", "Home")
            away = (meta.get("away_team") or {}).get("away_team_name", "Away")
            hs, a_s = meta.get("home_score", "?"), meta.get("away_score", "?")
            date = meta.get("match_date", "")
            return f"{home} {hs}-{a_s} {away} ({date})".strip()
        except Exception:
            pass
    return f"Match {match_id}"


def create_app() -> Flask:
    """Build the Flask app. (No .env read here — see main()/Cloud Run note above.)"""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        """Serve the single-page UI."""
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/languages")
    def languages():
        """List selectable languages (BCP-47 code + display name), sorted by name."""
        items = [{"code": c, "name": n} for c, n in prompts.LANGUAGE_NAMES.items()]
        items.sort(key=lambda x: x["name"])
        return jsonify(items)

    @app.get("/api/matches")
    def matches():
        """List the bundled sample plus any cached matches under data/cache/."""
        out = [{"id": "sample", "label": "Sample build-up (bundled, ~16 events)"}]
        if CACHE_DIR.exists():
            for entry in sorted(CACHE_DIR.iterdir()):
                if (entry / "events.json").exists():
                    out.append({"id": entry.name, "label": _match_label(entry.name)})
        return jsonify(out)

    @app.get("/api/stream")
    def stream():
        """Stream commentary as Server-Sent Events (one event per spoken line)."""
        match = request.args.get("match", "sample")
        language = request.args.get("language", os.getenv("DEFAULT_LANGUAGE", "en-US"))
        speed = float(request.args.get("speed", os.getenv("REPLAY_SPEED", "30")))
        mock = request.args.get("mock", "1") == "1"
        context_enabled = request.args.get("context", "0") == "1"
        tts_enabled = request.args.get("tts", "0") == "1"

        try:
            events = _load_events(match)
        except FileNotFoundError as exc:
            body = f"event: streamerror\ndata: {json.dumps(str(exc))}\n\n"
            return Response(body, mimetype="text/event-stream")

        def generate():
            try:
                for item in stream_commentary(
                    events,
                    language=language,
                    speed=speed,
                    mock=mock,
                    context_enabled=context_enabled,
                    tts_enabled=tts_enabled,
                    tts_provider="google" if tts_enabled else "noop",
                ):
                    payload = item.as_dict()
                    if item.speech.audio_path:
                        payload["audio_url"] = "/api/audio/" + Path(item.speech.audio_path).name
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as exc:  # surface, don't crash the worker
                yield f"event: streamerror\ndata: {json.dumps(str(exc))}\n\n"
            yield "event: done\ndata: {}\n\n"

        # X-Accel-Buffering off so proxies (incl. Cloud Run) flush each event.
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/audio/<path:filename>")
    def audio(filename):
        """Serve a generated mp3 from tts/out/ for synced playback."""
        return send_from_directory(DEFAULT_OUT_DIR, filename, mimetype="audio/mpeg")

    @app.get("/api/tts")
    def tts():
        """Synthesize a single line on demand (used for manual replay)."""
        text = request.args.get("text", "")
        language = request.args.get("language", "en-US")
        result = GoogleCloudSpeaker().synthesize(text, language=language)
        if not result.has_audio():
            return jsonify({"error": result.skipped_reason}), 503
        return Response(result.audio_bytes, mimetype=result.mime_type)

    return app


# Module-level app for WSGI servers (gunicorn web.app:app on Cloud Run).
app = create_app()


def main(argv=None) -> int:
    """
    Run the local dev server (loads .env for local convenience).

    Defaults to 127.0.0.1:8080. On Windows, 8080 is often taken or sits inside a
    reserved/excluded port range (Hyper-V / WSL / Docker), which raises
    "WinError 10013: An attempt was made to access a socket in a way forbidden by
    its access permissions". If that happens, just pick another port:
        python -m web.app --port 5050
    Use --host 0.0.0.0 to expose the server on your LAN (e.g. to view on a phone).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the MlangCast web UI.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"),
                        help="Bind address (default 127.0.0.1; use 0.0.0.0 for LAN).")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")),
                        help="Port to serve on (try 5050 or 8000 if 8080 is blocked).")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")  # local only; Cloud Run injects env vars directly
    except ImportError:
        pass

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"MlangCast UI -> http://{shown}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
