"""
context/player_stats.py  —  SCAFFOLD (Feature 1 data layer).

Builds the player-form context that the dead-air color commentary talks about.
Tournament form is derived by AGGREGATING each player's StatsBomb events across
their WC 2022 matches (same free data source as the play-by-play), optionally
blended with a few curated bio lines, and stored as kind:"player" documents in the
MongoDB `context` collection (via context.seed_context.seed_context_store).

SCOPE (demo): the two finalists' squads (~30+ players). The aggregation works over
ANY list of cached matches, so it generalises to more teams/competitions — state
that plainly in the README + video.

STATUS: aggregation + document building are implemented (pure, offline-testable);
the CLI can preview or seed the resulting player docs into MongoDB.

Wiring (you do this — this is a NEW file; no existing files were modified):
    forms = aggregate_player_form(load_events_by_match([...match ids...]))
    docs  = build_player_documents(forms)
    from context.seed_context import connect_collection, seed_context_store, embed_documents
    col = connect_collection()
    docs = [{**d, "embedding": v} for d, v in zip(docs, embed_documents([d["text"] for d in docs]))]
    seed_context_store(docs, client=col)     # now retrievable by MongoMCPContextClient
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "cache"

# A few curated one-liners to blend with computed form (optional "color").
# Keep them factual + low-risk; the agent must still not invent *stats*.
# TODO: extend to cover both full squads.
CURATED_BIOS: Dict[str, str] = {
    "Lionel Andrés Messi Cuccittini":
        "Argentina captain and talisman, celebrated for playmaking and dribbling.",
    "Kylian Mbappé Lottin":
        "France's explosive forward, feared for blistering pace and finishing.",
    "Ángel Fabián Di María Hernández":
        "Argentina winger with a habit of scoring in finals.",
    "Antoine Griezmann":
        "France's creative hub, linking midfield and attack.",
    "Emiliano Martínez":
        "Argentina goalkeeper, known as a penalty-shootout specialist.",
}


@dataclass
class PlayerForm:
    """Aggregated tournament tallies for one player (computed from events)."""

    name: str
    team: str = ""
    matches: int = 0
    goals: int = 0
    assists: int = 0
    key_passes: int = 0
    shots: int = 0
    dribbles: int = 0
    passes: int = 0

    def text(self, bio: str = "") -> str:
        """A faithful, compact form summary for embedding/retrieval."""
        bits = []
        if self.goals:
            bits.append(f"{self.goals} goal{'s' if self.goals != 1 else ''}")
        if self.assists:
            bits.append(f"{self.assists} assist{'s' if self.assists != 1 else ''}")
        if self.key_passes:
            bits.append(f"{self.key_passes} key passes")
        if self.shots:
            bits.append(f"{self.shots} shots")
        if self.dribbles:
            bits.append(f"{self.dribbles} completed dribbles")
        stat = ", ".join(bits) if bits else "limited end-product so far"
        team = f" ({self.team})" if self.team else ""
        line = f"{self.name}{team}: across {self.matches} WC2022 match(es) — {stat}."
        return f"{line} {bio}".strip() if bio else line


def _name(value) -> str:
    """Pull a nested StatsBomb {'name': ...} field safely."""
    return (value or {}).get("name", "") if isinstance(value, dict) else ""


# --------------------------------------------------------------------------- #
# Aggregation (the heart of the data layer — pure + offline-testable)         #
# --------------------------------------------------------------------------- #
def aggregate_player_form(events_by_match: Dict[object, List[dict]]) -> Dict[str, PlayerForm]:
    """
    Aggregate per-player tournament tallies from StatsBomb events.

    `events_by_match` maps any match key -> that match's event list. Counts goals
    (Shot + outcome Goal), assists (pass.goal_assist), key passes (pass.shot_assist),
    shots, completed dribbles, and passes; `matches` counts distinct appearances.
    """
    forms: Dict[str, PlayerForm] = {}
    for _match_key, events in events_by_match.items():
        appeared = set()
        for ev in events or []:
            player = _name(ev.get("player"))
            if not player:
                continue
            form = forms.setdefault(player, PlayerForm(name=player))
            if not form.team:
                form.team = _name(ev.get("team"))
            appeared.add(player)

            etype = _name(ev.get("type"))
            if etype == "Pass":
                form.passes += 1
                p = ev.get("pass") or {}
                if p.get("goal_assist"):
                    form.assists += 1
                if p.get("shot_assist"):
                    form.key_passes += 1
            elif etype == "Shot":
                form.shots += 1
                if _name((ev.get("shot") or {}).get("outcome")) == "Goal":
                    form.goals += 1
            elif etype == "Dribble":
                if _name((ev.get("dribble") or {}).get("outcome")) == "Complete":
                    form.dribbles += 1
        for player in appeared:
            forms[player].matches += 1
    return forms


def build_player_documents(
    forms: Dict[str, PlayerForm],
    bios: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Turn aggregated forms into kind:'player' context docs (same schema as seeds)."""
    bios = CURATED_BIOS if bios is None else bios
    return [
        {
            "kind": "player",
            "name": name,
            "team": form.team,
            "text": form.text(bios.get(name, "")),
        }
        for name, form in sorted(forms.items())
    ]


# --------------------------------------------------------------------------- #
# Match discovery / loading (thin; network parts are stubbed)                 #
# --------------------------------------------------------------------------- #
def load_events_by_match(match_ids: Iterable[int]) -> Dict[int, List[dict]]:
    """Load cached events for each match id (silently skips ones not downloaded)."""
    out: Dict[int, List[dict]] = {}
    for mid in match_ids:
        path = CACHE_DIR / str(mid) / "events.json"
        if path.exists():
            out[mid] = json.loads(path.read_text(encoding="utf-8"))
    return out


def discover_team_match_ids(
    team_names: Iterable[str],
    competition_id: int = 43,
    season_id: int = 106,
) -> List[int]:
    """
    Find WC2022 match ids involving any of `team_names` (e.g. the two finalists).

    TODO: this hits the network via data.loader.list_matches — add caching/retry
    to taste. Used by the --teams CLI path.
    """
    from data.loader import list_matches  # lazy: needs network

    wanted = {t.lower() for t in team_names}
    ids: List[int] = []
    for m in list_matches(competition_id, season_id):
        home = (m.get("home_team") or {}).get("home_team_name", "").lower()
        away = (m.get("away_team") or {}).get("away_team_name", "").lower()
        if home in wanted or away in wanted:
            ids.append(m["match_id"])
    return ids


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """Preview player-form docs from cached matches (offline by default)."""
    parser = argparse.ArgumentParser(description="Build WC2022 player-form context docs.")
    parser.add_argument("--match-ids", type=int, nargs="*", default=[3869685],
                        help="Cached match ids to aggregate (default: the final).")
    parser.add_argument("--teams", nargs="*", default=None,
                        help="Discover all WC2022 matches for these teams (needs network).")
    parser.add_argument("--download", action="store_true",
                        help="Download any missing matches before aggregating (needs network).")
    parser.add_argument("--seed", action="store_true",
                        help="Write player docs into MongoDB context collection.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Seed without Gemini embeddings (text-search only).")
    parser.add_argument("--replace", action="store_true",
                        help="Clear the collection before inserting player docs.")
    args = parser.parse_args(argv)

    match_ids = list(args.match_ids)
    if args.teams:
        match_ids = discover_team_match_ids(args.teams)

    if args.download:
        from data.loader import cache_match  # lazy: needs network
        for mid in match_ids:
            if not (CACHE_DIR / str(mid) / "events.json").exists():
                cache_match(mid)

    forms = aggregate_player_form(load_events_by_match(match_ids))
    docs = build_player_documents(forms)
    if not args.seed:
        print(json.dumps(docs, indent=2, ensure_ascii=False))
        print(f"\n# {len(docs)} player docs from {len(match_ids)} cached match(es).")
        print("# Add --seed to write these kind:'player' docs into MongoDB context.")
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass

    from context.seed_context import connect_collection, embed_documents, seed_context_store

    collection = connect_collection()
    if not args.no_embed:
        vectors = embed_documents([doc["text"] for doc in docs])
        docs = [{**doc, "embedding": vec} for doc, vec in zip(docs, vectors)]
    status = seed_context_store(docs, client=collection, replace=args.replace)
    print(f"Seeded {status['inserted']} player docs into MongoDB context "
          f"({'without' if args.no_embed else 'with'} embeddings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
