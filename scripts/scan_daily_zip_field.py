#!/usr/bin/env python3
"""scan_daily_zip_field.py — build a raw games.csv from Kaggle daily episode-dump
zips (data/kaggle_daily_zips/*.zip), reading zip entries straight into memory
(zipfile .namelist()/.read()) — the ~20GB uncompressed content of one zip is
NEVER extracted to disk.

Reuses existing project code as-is (no logic duplicated):
  - src/replays/parse.py::true_decks_from_data   — each side's 60-card deck from
    steps[1][idx].action (already the ground truth used by
    scripts/scan_daily_dataset_for_experts.py and the topdeck/crustle recon tools).
  - scripts/classify_opponent_archetype.py::classify — the single archetype
    source of truth; called exactly as-is, nothing in it modified.
  - scripts/cluster_archetypes.py::load_card_meta — id -> card name lookup,
    needed because true_decks_from_data only returns card IDs, and classify()
    also wants the card NAMES (for its NAME_KEYWORDS fallback table).
  - scripts/gate_scout_guard.py::wilson_ci and
    scripts/cross_version_matchup_ledger.py::mantel_haenszel — imported here
    ONLY to prove the import path is clean for the follow-up step (0.4, which
    consumes games.csv); neither is called by this script. Plain `from module
    import name` works fine — both files guard their argparse/side-effecting
    code under `if __name__ == "__main__"`, so importing them under the
    project's .venv (3.11) has no side effects. (A bare `python3` on this
    machine is 3.9 and gate_scout_guard's `import cg_native` chain uses
    `X | None` unions that don't parse on 3.9 — that's a Python-version issue,
    not an import-side-effect issue. Run this script with .venv/bin/python.)

Rating caveat (found by inspection — see also the report to the user):
  Neither manifest.csv nor the replay JSON carries a per-side rating.
    - manifest.csv columns are exactly:
      episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes
      avg_score/min_score/sum_score are aggregated over BOTH agents in the
      episode (sum_score = 2*avg_score; min_score = min of the two sides) —
      there's no way to recover which side had which rating from this alone.
    - The replay JSON's info.Agents[i] only has {Name, ThumbnailUrl}; no
      team_id, no rating field anywhere in the episode JSON.
  The only per-side identity available in the daily-zip data is the agent's
  display Name (== TeamNames[i]). So rating_a/rating_b here are filled by
  joining team_a/team_b against the freshest local public-leaderboard CSV
  snapshot found under out/**, docs/gold/, scratchpad/gold/ (TeamName -> Score
  columns) — the same file format scripts/classify_opponent_archetype.py's
  load_leaderboard_scores() already reads, just keyed by TeamName instead of
  TeamId since TeamId isn't available here. This is a CURRENT rating (as of
  the leaderboard snapshot), not the rating at the exact moment the game was
  played — the best available proxy given what's actually in this dataset.
  Teams not present in that snapshot get a blank rating (counted, not
  silently guessed).

Usage:
    .venv/bin/python scripts/scan_daily_zip_field.py <out_dir> [zip_or_dir ...] [--limit N]
    # zip_or_dir defaults to data/kaggle_daily_zips/ (all *.zip in it)

Output: <out_dir>/games.csv with columns exactly:
    episode_id,team_a,arche_a,rating_a,team_b,arche_b,rating_b,winner
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from replays.parse import true_decks_from_data  # noqa: E402
from classify_opponent_archetype import classify  # noqa: E402
from cluster_archetypes import load_card_meta  # noqa: E402

# Import-only: proves the path works for the follow-up step (0.4). Not called here.
from gate_scout_guard import wilson_ci  # noqa: E402,F401
from cross_version_matchup_ledger import mantel_haenszel  # noqa: E402,F401

FIELDNAMES = ["episode_id", "team_a", "arche_a", "rating_a", "team_b", "arche_b", "rating_b", "winner"]

# How many full tracebacks to print per skip-reason bucket, so a systematic bug
# is visible without flooding stdout for 13k episodes.
MAX_TRACEBACKS_PER_REASON = 3


def find_leaderboard_csvs() -> list[Path]:
    patterns = [
        REPO / "out",
        REPO / "docs" / "gold",
        REPO / "scratchpad" / "gold",
    ]
    found = []
    for base in patterns:
        if base.is_dir():
            found.extend(base.glob("**/*leaderboard*.csv"))
    return found


def load_leaderboard_by_name() -> tuple[dict[str, float], Path | None]:
    """{TeamName: Score} from the freshest local public-leaderboard CSV snapshot.
    Returns ({}, None) if none is found (rating_a/rating_b will be left blank)."""
    files = find_leaderboard_csvs()
    if not files:
        return {}, None
    path = max(files, key=lambda p: p.stat().st_mtime)
    scores: dict[str, float] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get("TeamName")
            try:
                score = float(row["Score"])
            except (KeyError, TypeError, ValueError):
                continue
            if name:
                scores[name] = score
    return scores, path


class SkipStats:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self._tb_shown: dict[str, int] = {}
        self.ok = 0

    def skip(self, reason: str, entry_name: str = "", exc: BaseException | None = None):
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if exc is not None:
            shown = self._tb_shown.get(reason, 0)
            if shown < MAX_TRACEBACKS_PER_REASON:
                print(f"  [skip:{reason}] {entry_name}: {exc!r}")
                traceback.print_exc()
                self._tb_shown[reason] = shown + 1

    def total_skipped(self) -> int:
        return sum(self.counts.values())


def process_episode(raw: bytes, entry_name: str, meta, name_to_rating: dict[str, float],
                     stats: SkipStats, seen_episode_ids: set) -> dict | None:
    try:
        data = json.loads(raw)
    except Exception as e:
        stats.skip("json_parse_error", entry_name, e)
        return None

    try:
        info = data.get("info", {}) or {}
        agents = info.get("Agents", []) or []
        rewards = data.get("rewards", []) or []
        episode_id = info.get("EpisodeId") or data.get("id")

        if len(agents) < 2 or len(rewards) < 2:
            stats.skip("bad_agents_or_rewards", entry_name)
            return None

        if episode_id in seen_episode_ids:
            stats.skip("duplicate_episode_id", entry_name)
            return None

        decks = true_decks_from_data(data)
        if 0 not in decks or 1 not in decks:
            stats.skip("deck_not_extractable", entry_name)
            return None

        team_a = agents[0].get("Name", "") or ""
        team_b = agents[1].get("Name", "") or ""

        names_a = [meta.name.get(cid, "") for cid in decks[0]]
        names_b = [meta.name.get(cid, "") for cid in decks[1]]
        arche_a = classify(decks[0], names_a) or "UNKNOWN"
        arche_b = classify(decks[1], names_b) or "UNKNOWN"

        if rewards[0] > 0:
            winner = "team_a"
        elif rewards[1] > 0:
            winner = "team_b"
        else:
            stats.skip("no_clear_winner", entry_name)
            return None

        seen_episode_ids.add(episode_id)
        stats.ok += 1
        return {
            "episode_id": episode_id,
            "team_a": team_a,
            "arche_a": arche_a,
            "rating_a": name_to_rating.get(team_a, ""),
            "team_b": team_b,
            "arche_b": arche_b,
            "rating_b": name_to_rating.get(team_b, ""),
            "winner": winner,
        }
    except Exception as e:
        stats.skip("other_error", entry_name, e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("zips", nargs="*", default=[],
                     help="zip files or a directory of zips; defaults to data/kaggle_daily_zips/")
    ap.add_argument("--limit", type=int, default=0, help="debug only: cap episodes per zip (0 = no limit)")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    if args.zips:
        paths = [Path(p) for p in args.zips]
    else:
        paths = [REPO / "data" / "kaggle_daily_zips"]

    zip_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            zip_paths.extend(sorted(p.glob("*.zip")))
        elif p.suffix == ".zip":
            zip_paths.append(p)
    if not zip_paths:
        sys.exit(f"No zip files found in {paths}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "games.csv"

    print(f"[scan] {len(zip_paths)} zip(s): {[p.name for p in zip_paths]}")

    meta = load_card_meta()
    name_to_rating, leaderboard_path = load_leaderboard_by_name()
    if leaderboard_path:
        print(f"[scan] rating source: {leaderboard_path} ({len(name_to_rating)} teams)")
    else:
        print("[scan] WARNING: no local *leaderboard*.csv found — rating_a/rating_b will be blank for all rows")

    stats = SkipStats()
    seen_episode_ids: set = set()

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        for zp in zip_paths:
            print(f"[scan] === {zp.name} ===")
            with zipfile.ZipFile(zp) as zf:
                names = [n for n in zf.namelist() if n.endswith(".json")]
                if args.limit:
                    names = names[: args.limit]
                print(f"[scan]   {len(names)} episode JSON entries")
                for i, entry_name in enumerate(names, 1):
                    try:
                        raw = zf.read(entry_name)
                    except Exception as e:
                        stats.skip("zip_read_error", entry_name, e)
                        continue
                    row = process_episode(raw, entry_name, meta, name_to_rating, stats, seen_episode_ids)
                    if row is not None:
                        writer.writerow(row)
                    if i % args.progress_every == 0 or i == len(names):
                        print(f"[scan]   {zp.name}: {i}/{len(names)} entries "
                              f"(ok={stats.ok}, skipped={stats.total_skipped()})")

    print("\n=== DONE ===")
    print(f"Wrote {out_csv}")
    print(f"Processed OK: {stats.ok}")
    print(f"Skipped total: {stats.total_skipped()}")
    for reason, count in sorted(stats.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
