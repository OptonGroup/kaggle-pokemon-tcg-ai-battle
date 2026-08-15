#!/usr/bin/env python3
"""scan_mirror_fan_field.py — B1 (MIRROR_LEVERS_BRIEF рычаг B): пересчитать WR
зеркала Grimmsnarl-vs-Grimmsnarl "с Handheld Fan (id 1161) vs без него" на
свежем окне daily-zip дампа эпизодов (data/kaggle_daily_zips/*2026-08-0[1-9]*,
*2026-08-10*).

Модель по образцу scripts/scan_daily_zip_field.py: читает zip-записи прямо в
память (без распаковки на диск), деки берёт из src/replays/parse.py::
true_decks_from_data (steps[1][idx].action — та же истина, что и у остального
проекта), архетип — scripts/classify_opponent_archetype.py::classify (единый
источник правды).

Отбор: обе стороны arche == mirror_grimmsnarl (ключевые карты {646,647,648}
по ARCHETYPE_IDS). Для каждой такой игры пишем строку в games.csv с обеими
деками, флагами has_fan_a/has_fan_b (card id 1161 в 60-card листе), рейтингами
(join по TeamName к локальному снапшоту паблик-лидерборда — как и в
scan_daily_zip_field.py; это ТЕКУЩИЙ рейтинг команды на момент снапшота, не
рейтинг в момент конкретной игры — тот же уже задокументированный компромис).

Usage:
    .venv/bin/python scripts/scan_mirror_fan_field.py <out_dir> [zip_or_dir ...]
        [--leaderboard PATH] [--limit N]
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
from classify_opponent_archetype import classify, ARCHETYPE_IDS  # noqa: E402
from cluster_archetypes import load_card_meta  # noqa: E402

FAN_ID = 1161
MIRROR_ARCHE = "mirror_grimmsnarl"
assert MIRROR_ARCHE in ARCHETYPE_IDS

FIELDNAMES = [
    "episode_id", "day",
    "team_a", "rating_a", "has_fan_a",
    "team_b", "rating_b", "has_fan_b",
    "winner",  # "team_a" | "team_b"
]


def find_leaderboard_csvs() -> list[Path]:
    found = []
    for base in (REPO / "out", REPO / "docs" / "gold", REPO / "scratchpad"):
        if base.is_dir():
            found.extend(base.glob("**/*leaderboard*.csv"))
    return found


def load_leaderboard_by_name(explicit: str | None) -> tuple[dict[str, float], Path | None]:
    if explicit:
        path = Path(explicit)
    else:
        files = find_leaderboard_csvs()
        if not files:
            return {}, None
        # Prefer a snapshot dated inside the 08-01..08-10 window if one exists,
        # else fall back to the freshest overall (closer date = better proxy).
        window = [p for p in files if "2026-08-0" in p.name or "2026-08-10" in p.name]
        pool = window or files
        path = max(pool, key=lambda p: p.stat().st_mtime)
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
        self.mirror_games = 0

    def skip(self, reason: str, entry_name: str = "", exc: BaseException | None = None):
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if exc is not None:
            shown = self._tb_shown.get(reason, 0)
            if shown < 3:
                print(f"  [skip:{reason}] {entry_name}: {exc!r}")
                traceback.print_exc()
                self._tb_shown[reason] = shown + 1


def process_episode(raw: bytes, entry_name: str, day: str, meta, name_to_rating: dict[str, float],
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

        names_a = [meta.name.get(cid, "") for cid in decks[0]]
        names_b = [meta.name.get(cid, "") for cid in decks[1]]
        arche_a = classify(decks[0], names_a) or "UNKNOWN"
        arche_b = classify(decks[1], names_b) or "UNKNOWN"

        stats.ok += 1
        if arche_a != MIRROR_ARCHE or arche_b != MIRROR_ARCHE:
            stats.skip("not_mirror")
            return None

        if rewards[0] > 0:
            winner = "team_a"
        elif rewards[1] > 0:
            winner = "team_b"
        else:
            stats.skip("no_clear_winner", entry_name)
            return None

        seen_episode_ids.add(episode_id)
        stats.mirror_games += 1

        team_a = agents[0].get("Name", "") or ""
        team_b = agents[1].get("Name", "") or ""

        return {
            "episode_id": episode_id,
            "day": day,
            "team_a": team_a,
            "rating_a": name_to_rating.get(team_a, ""),
            "has_fan_a": int(FAN_ID in decks[0]),
            "team_b": team_b,
            "rating_b": name_to_rating.get(team_b, ""),
            "has_fan_b": int(FAN_ID in decks[1]),
            "winner": winner,
        }
    except Exception as e:
        stats.skip("other_error", entry_name, e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("zips", nargs="*", default=[])
    ap.add_argument("--leaderboard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    if args.zips:
        paths = [Path(p) for p in args.zips]
    else:
        paths = [REPO / "data" / "kaggle_daily_zips"]

    zip_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            zip_paths.extend(sorted(p.glob("pokemon-tcg-ai-battle-episodes-2026-08-*.zip")))
        elif p.suffix == ".zip":
            zip_paths.append(p)
    # restrict strictly to the 08-01..08-10 window even if a wider dir is passed
    zip_paths = [p for p in zip_paths if any(f"2026-08-{d:02d}" in p.name for d in range(1, 11))]
    if not zip_paths:
        sys.exit(f"No zip files found in window 08-01..08-10 under {paths}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mirror_games.csv"

    print(f"[scan] {len(zip_paths)} zip(s): {[p.name for p in zip_paths]}")

    meta = load_card_meta()
    name_to_rating, leaderboard_path = load_leaderboard_by_name(args.leaderboard or None)
    if leaderboard_path:
        print(f"[scan] rating source: {leaderboard_path} ({len(name_to_rating)} teams)")
    else:
        print("[scan] WARNING: no local *leaderboard*.csv found — ratings blank for all rows")

    stats = SkipStats()
    seen_episode_ids: set = set()

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        for zp in zip_paths:
            day = zp.stem.rsplit("-", 3)[-3:]  # crude, overwritten below properly
            # filename: pokemon-tcg-ai-battle-episodes-2026-08-05 -> day = 2026-08-05
            day = zp.stem.split("episodes-")[-1]
            print(f"[scan] === {zp.name} (day={day}) ===")
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
                    row = process_episode(raw, entry_name, day, meta, name_to_rating, stats, seen_episode_ids)
                    if row is not None:
                        writer.writerow(row)
                    if i % args.progress_every == 0 or i == len(names):
                        print(f"[scan]   {zp.name}: {i}/{len(names)} entries "
                              f"(ok={stats.ok}, mirror={stats.mirror_games}, skipped={stats.total_skipped() if hasattr(stats,'total_skipped') else sum(stats.counts.values())})")

    print("\n=== DONE ===")
    print(f"Wrote {out_csv}")
    print(f"Total episodes parsed OK: {stats.ok}")
    print(f"Mirror grimmsnarl games: {stats.mirror_games}")
    for reason, count in sorted(stats.counts.items(), key=lambda kv: -kv[1]):
        print(f"  skip:{reason}: {count}")


if __name__ == "__main__":
    main()
