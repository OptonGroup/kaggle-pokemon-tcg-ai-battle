#!/usr/bin/env python3
"""build_candidate_decks.py — Phase 1: for each candidate archetype (lopunny,
ogerpon_teal), collect every WINNING full 60-card decklist played by that
archetype across the three daily zips, so a single "best" decklist can be
picked per archetype (see scripts/pick_candidate_decklist.py).

Reuses the exact same scan pattern as scripts/scan_daily_zip_field.py:
  - true_decks_from_data (src/replays/parse.py) for the ground-truth 60-card
    deck per side.
  - classify() (scripts/classify_opponent_archetype.py) — used here only as a
    sanity re-check against out/field_recon_20260729_0801/games.csv's already
    -computed arche_a/arche_b (Phase 0 ran classify() on every episode; we
    trust that cache to select which episodes/sides to look at, instead of
    reclassifying 13k games — but every extracted deck is reclassified and
    compared against the cached label, so a games.csv/zip mismatch or a
    dedup/printing bug would surface immediately as a loud error, not a
    silent bad decklist).
  - load_card_meta (scripts/cluster_archetypes.py) for card names (needed by
    classify()'s NAME_KEYWORDS fallback).
  - find_leaderboard_csvs / load_leaderboard_by_name (scripts/scan_daily_zip_field.py)
    for the exact same team_name -> rating join used in games.csv.

Only episodes whose games.csv row has arche_a or arche_b == one of the target
archetypes are opened (matched directly by zip entry name == f"{episode_id}.json"),
so this reads a small fraction of the ~13k episodes across the 3 zips, not all
of them.

For each target archetype, ALSO tracks win/loss counts per exact-deck-signature
(multiset of 60 card ids) across every episode that archetype played (not just
wins) — needed for the raw within-archetype WR sanity check in the picking step.

Output:
  out/field_recon_20260729_0801/candidate_decks/{archetype}.jsonl
    one JSON object per WINNING game: {"episode_id", "team_name", "deck_ids"}
  out/field_recon_20260729_0801/candidate_decks/{archetype}_sig_wl.json
    {signature_key: {"wins": int, "losses": int}} where signature_key is a
    "cid:count,cid:count,..." string (sorted by cid) — the raw WR denominator
    data, over ALL episodes (win + loss), not just the saved winners.

Usage:
    .venv/bin/python scripts/build_candidate_decks.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from replays.parse import true_decks_from_data  # noqa: E402
from classify_opponent_archetype import classify  # noqa: E402
from cluster_archetypes import load_card_meta  # noqa: E402
from scan_daily_zip_field import load_leaderboard_by_name  # noqa: E402

TARGET_ARCHETYPES = ["lopunny", "ogerpon_teal"]
GAMES_CSV = REPO / "out" / "field_recon_20260729_0801" / "games.csv"
OUT_DIR = REPO / "out" / "field_recon_20260729_0801" / "candidate_decks"
ZIP_DIR = REPO / "data" / "kaggle_daily_zips"


def sig_key(deck_ids: list[int]) -> str:
    """Deterministic string key for a multiset-of-60-ids signature."""
    return ",".join(f"{cid}:{cnt}" for cid, cnt in sorted(Counter(deck_ids).items()))


def load_needed_episodes() -> dict[str, dict[str, tuple[str, str, bool]]]:
    """Returns {episode_id_str: {side ('a'|'b'): (archetype, team_name, is_win)}}
    for every (episode, side) whose games.csv row's arche_<side> is one of the
    target archetypes."""
    needed: dict[str, dict[str, tuple[str, str, bool]]] = {}
    with open(GAMES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for side, arche_key, team_key, win_val in [
                ("a", "arche_a", "team_a", "team_a"),
                ("b", "arche_b", "team_b", "team_b"),
            ]:
                arche = row[arche_key]
                if arche in TARGET_ARCHETYPES:
                    is_win = row["winner"] == win_val
                    needed.setdefault(row["episode_id"], {})[side] = (
                        arche, row[team_key], is_win,
                    )
    return needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                     help="debug only: cap needed-episode lookups per zip (0 = no limit)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    needed = load_needed_episodes()
    print(f"[build] needed episodes (either side matches a target archetype): {len(needed)}")
    per_arche_needed = Counter()
    for sides in needed.values():
        for arche, _, is_win in sides.values():
            per_arche_needed[arche] += 1
    print(f"[build] per-archetype episode-sides needed: {dict(per_arche_needed)}")

    meta = load_card_meta()
    name_to_rating, leaderboard_path = load_leaderboard_by_name()
    print(f"[build] rating source: {leaderboard_path} ({len(name_to_rating)} teams)")

    zip_paths = sorted(ZIP_DIR.glob("*.zip"))
    print(f"[build] {len(zip_paths)} zip(s): {[p.name for p in zip_paths]}")

    # accumulators
    win_records: dict[str, list[dict]] = {a: [] for a in TARGET_ARCHETYPES}
    sig_wl: dict[str, dict[str, Counter]] = {a: {} for a in TARGET_ARCHETYPES}
    mismatches = []
    found_episode_ids: set[str] = set()
    deck_not_extractable = 0
    zip_read_errors = 0

    for zp in zip_paths:
        print(f"[build] === {zp.name} ===")
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            hits = [n for n in names if n[:-5] in needed and n.endswith(".json")]
            if args.limit:
                hits = hits[: args.limit]
            print(f"[build]   {len(hits)} matching entries in this zip")
            for i, entry_name in enumerate(hits, 1):
                ep_id = entry_name[:-5]
                if ep_id in found_episode_ids:
                    continue  # already satisfied from an earlier zip (shouldn't happen, episode_ids are unique, but be safe)
                try:
                    raw = zf.read(entry_name)
                    data = json.loads(raw)
                except Exception as e:
                    zip_read_errors += 1
                    print(f"  [skip:zip_read_error] {entry_name}: {e!r}")
                    continue

                decks = true_decks_from_data(data)
                side_idx = {"a": 0, "b": 1}
                for side, (arche, team_name, is_win) in needed[ep_id].items():
                    idx = side_idx[side]
                    if idx not in decks:
                        deck_not_extractable += 1
                        print(f"  [skip:deck_not_extractable] ep={ep_id} side={side} arche={arche}")
                        continue
                    deck_ids = decks[idx]

                    # sanity re-check against the games.csv cached label
                    names_list = [meta.name.get(cid, "") for cid in deck_ids]
                    reclassified = classify(deck_ids, names_list) or "UNKNOWN"
                    if reclassified != arche:
                        mismatches.append((ep_id, side, arche, reclassified))
                        print(f"  [MISMATCH] ep={ep_id} side={side} games.csv={arche} "
                              f"reclassify={reclassified} -- skipping this (episode,side)")
                        continue

                    key = sig_key(deck_ids)
                    bucket = sig_wl[arche].setdefault(key, Counter())
                    bucket["wins" if is_win else "losses"] += 1

                    if is_win:
                        win_records[arche].append({
                            "episode_id": ep_id,
                            "team_name": team_name,
                            "deck_ids": deck_ids,
                        })
                found_episode_ids.add(ep_id)
                if i % 200 == 0 or i == len(hits):
                    print(f"[build]   {zp.name}: {i}/{len(hits)} matched entries processed")

    missing = set(needed) - found_episode_ids
    if missing:
        print(f"[build] WARNING: {len(missing)} needed episode_ids not found in any zip: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    if mismatches:
        print(f"\n[build] WARNING: {len(mismatches)} games.csv/reclassify mismatches (see above), excluded from output")

    print(f"\n[build] deck_not_extractable: {deck_not_extractable}  zip_read_errors: {zip_read_errors}")

    for arche in TARGET_ARCHETYPES:
        out_jsonl = OUT_DIR / f"{arche}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in win_records[arche]:
                f.write(json.dumps(rec) + "\n")
        print(f"[build] wrote {out_jsonl}: {len(win_records[arche])} winning games")

        out_wl = OUT_DIR / f"{arche}_sig_wl.json"
        wl_serializable = {k: dict(v) for k, v in sig_wl[arche].items()}
        with open(out_wl, "w", encoding="utf-8") as f:
            json.dump(wl_serializable, f, indent=1)
        n_sigs = len(wl_serializable)
        total_wins = sum(v.get("wins", 0) for v in wl_serializable.values())
        total_losses = sum(v.get("losses", 0) for v in wl_serializable.values())
        print(f"[build] wrote {out_wl}: {n_sigs} unique signatures, "
              f"{total_wins} wins + {total_losses} losses tracked")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
