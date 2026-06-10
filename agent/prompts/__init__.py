"""
agent/prompts  —  prompt assembly for the commentary agent

The system prompt is composed from four editable building blocks so each concern
can be tuned independently

    faithfulness.md  — the "no inventing events" guardrail (never cut)
    pacing.md        — when to speak vs. stay quiet
    energy.md        — matching tone to the moment
    language.md      — generate natively in the target language

`build_event_prompt` turns a single event (+ match state + retrieved context) into
the per-turn user message.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_DIR = Path(__file__).resolve().parent

LANGUAGE_NAMES = {
    "af-ZA": "Afrikaans (South Africa)", "ar-XA": "Arabic",
    "eu-ES": "Basque (Spain)", "bn-IN": "Bengali (India)",
    "bg-BG": "Bulgarian (Bulgaria)", "ca-ES": "Catalan (Spain)",
    "yue-HK": "Chinese, Cantonese (Hong Kong)", "cmn-CN": "Chinese, Mandarin (China)",
    "cmn-TW": "Chinese, Mandarin (Taiwan)", "hr-HR": "Croatian (Croatia)",
    "cs-CZ": "Czech (Czechia)", "da-DK": "Danish (Denmark)",
    "nl-BE": "Dutch (Belgium)", "nl-NL": "Dutch (Netherlands)",
    "en-AU": "English (Australia)", "en-IN": "English (India)",
    "en-GB": "English (United Kingdom)", "en-US": "English (United States)",
    "fil-PH": "Filipino (Philippines)", "fi-FI": "Finnish (Finland)",
    "fr-CA": "French (Canada)", "fr-FR": "French (France)",
    "gl-ES": "Galician (Spain)", "de-DE": "German (Germany)",
    "el-GR": "Greek (Greece)", "gu-IN": "Gujarati (India)",
    "he-IL": "Hebrew (Israel)", "hi-IN": "Hindi (India)",
    "hu-HU": "Hungarian (Hungary)", "is-IS": "Icelandic (Iceland)",
    "id-ID": "Indonesian (Indonesia)", "it-IT": "Italian (Italy)",
    "ja-JP": "Japanese (Japan)", "kn-IN": "Kannada (India)",
    "ko-KR": "Korean (South Korea)", "lv-LV": "Latvian (Latvia)",
    "lt-LT": "Lithuanian (Lithuania)", "ms-MY": "Malay (Malaysia)",
    "ml-IN": "Malayalam (India)", "mr-IN": "Marathi (India)",
    "nb-NO": "Norwegian, Bokmål (Norway)", "pl-PL": "Polish (Poland)",
    "pt-BR": "Portuguese (Brazil)", "pt-PT": "Portuguese (Portugal)",
    "pa-IN": "Punjabi (India)", "ro-RO": "Romanian (Romania)",
    "ru-RU": "Russian (Russia)", "sr-RS": "Serbian (Serbia)",
    "sk-SK": "Slovak (Slovakia)", "es-ES": "Spanish (Spain)",
    "es-US": "Spanish (United States)", "sv-SE": "Swedish (Sweden)",
    "ta-IN": "Tamil (India)", "te-IN": "Telugu (India)",
    "th-TH": "Thai (Thailand)", "tr-TR": "Turkish (Türkiye)",
    "uk-UA": "Ukrainian (Ukraine)", "vi-VN": "Vietnamese (Vietnam)",
}

LANGUAGE_ALIASES = {
    **{code: code for code in LANGUAGE_NAMES},
    "af": "af-ZA",
    "ar": "ar-XA",
    "eu": "eu-ES",
    "bn": "bn-IN",
    "bg": "bg-BG",
    "ca": "ca-ES",
    "yue": "yue-HK",
    "cmn": "cmn-CN",
    "zh": "cmn-CN",
    "zh-CN": "cmn-CN",
    "zh-TW": "cmn-TW",
    "zh-HK": "yue-HK",
    "hr": "hr-HR",
    "cs": "cs-CZ",
    "da": "da-DK",
    "nl": "nl-NL",
    "en": "en-US",
    "fil": "fil-PH",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "gl": "gl-ES",
    "de": "de-DE",
    "el": "el-GR",
    "gu": "gu-IN",
    "he": "he-IL",
    "hi": "hi-IN",
    "hu": "hu-HU",
    "is": "is-IS",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "kn": "kn-IN",
    "ko": "ko-KR",
    "lv": "lv-LV",
    "lt": "lt-LT",
    "ms": "ms-MY",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "nb": "nb-NO",
    "pl": "pl-PL",
    "pt": "pt-BR",
    "pa": "pa-IN",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sr": "sr-RS",
    "sk": "sk-SK",
    "es": "es-ES",
    "sv": "sv-SE",
    "ta": "ta-IN",
    "te": "te-IN",
    "th": "th-TH",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "vi": "vi-VN",
}

SUPPORTED_LANGUAGE_CODES = tuple(sorted({*LANGUAGE_NAMES, *LANGUAGE_ALIASES}))

_ROLE = (
    "You are MlangCast, a live football commentator generating spoken play-by-play "
    "for a match that would otherwise have no commentary at all."
    " Your goal is not to replace human commentary, but to fill in the gaps for "
    "lower-profile matches that would otherwise go unnoticed."
)


def _read(name: str) -> str:
    """Read one prompt block from disk so system_prompt can compose it."""
    return (_DIR / name).read_text(encoding="utf-8").strip()


def normalize_language(language: str = "en") -> str:
    """Return the canonical language code used internally by the agent."""
    return LANGUAGE_ALIASES.get(language, language)


def language_display_name(language: str = "en") -> str:
    """Return a human-readable language name for aliases and full BCP-47 codes."""
    normalized = normalize_language(language)
    return LANGUAGE_NAMES.get(normalized, language)


def system_prompt(language: str = "en") -> str:
    """Compose the full system instruction for a given target language."""
    lang_name = language_display_name(language)
    blocks = [
        _ROLE,
        _read("faithfulness.md"),
        _read("pacing.md"),
        _read("energy.md"),
        _read("language.md").replace("{language_name}", lang_name),
    ]
    return "\n\n---\n\n".join(blocks)


def _describe_event(ev: dict) -> str:
    """Faithful, compact rendering of the event the agent must comment on."""
    etype = ev.get("type", {}).get("name", "?")
    team = ev.get("team", {}).get("name", "?")
    player = (ev.get("player") or {}).get("name", "")
    parts = [f"type={etype}", f"team={team}"]
    if player:
        parts.append(f"player={player}")

    if etype == "Pass" and "pass" in ev:
        p = ev["pass"]
        if p.get("recipient"):
            parts.append(f"to={p['recipient'].get('name')}")
        if p.get("outcome"):
            parts.append(f"outcome={p['outcome'].get('name')}")
        if p.get("technique"):
            parts.append(f"technique={p['technique'].get('name')}")
    elif etype == "Shot" and "shot" in ev:
        s = ev["shot"]
        parts.append(f"outcome={(s.get('outcome') or {}).get('name')}")
        if s.get("body_part"):
            parts.append(f"body_part={s['body_part'].get('name')}")
        if s.get("statsbomb_xg") is not None:
            parts.append(f"xg={s['statsbomb_xg']:.2f}")
    elif etype in ("Foul Committed", "Bad Behaviour"):
        card = (ev.get("foul_committed", {}) or {}).get("card") \
            or (ev.get("bad_behaviour", {}) or {}).get("card")
        if card:
            parts.append(f"card={card.get('name')}")
    elif etype == "Substitution" and "substitution" in ev:
        repl = (ev["substitution"].get("replacement") or {}).get("name")
        if repl:
            parts.append(f"replacement={repl}")

    return ", ".join(parts)


def build_event_prompt(
    event: dict,
    state: Optional[dict] = None,
    context: Optional[dict] = None,
) -> str:
    """
    Per-turn user message: the current event, the match state, and any retrieved
    context. Faithfulness rules live in the system prompt.
    """
    state = state or {}
    lines = []
    if state:
        lines.append(f"MATCH STATE: {json.dumps(state, ensure_ascii=False)}")
    lines.append(f"CURRENT EVENT: {_describe_event(event)}")
    if context:
        lines.append(f"CONTEXT (retrieved, optional): {json.dumps(context, ensure_ascii=False)}")
    lines.append(
        "Give the next line of live commentary for the current event, or reply with "
        "exactly NO_COMMENT if this moment isn't worth saying anything about."
    )
    return "\n".join(lines)
