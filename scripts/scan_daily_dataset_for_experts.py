#!/usr/bin/env python3
"""scan_daily_dataset_for_experts.py — mine winner-side IL pairs for jiatu.l/bono
from bulk-downloaded daily episode datasets (data/kaggle_daily/{day}/*.json).

Fast ASCII grep pre-filter on the raw JSON text (names are plain ASCII, no
\\uXXXX escaping needed) before paying the json.load() cost, mirroring the
Majkel1337 job template (data/deck_scan_maj_pool/kernel_log.json).

Purity filter: the 60-card deck actually submitted by the expert in the
matched episode (steps[1][player_idx].action) must overlap the reference
top10_decks.json list for that name by >=54/60 (multiset intersection).
Episodes failing this are dropped and counted separately (stability report).

Usage:
    python3 scripts/scan_daily_dataset_for_experts.py \
        --days 2026-07-16 2026-07-17 2026-07-18 2026-07-19 \
        --targets jiatu.l bono \
        --out-dir data/il_grimmsnarl_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from replays.parse import parse_replay, true_decks_from_data  # noqa: E402


def multiset_overlap(a: list[int], b: list[int]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=["jiatu.l", "bono"])
    ap.add_argument("--daily-root", default=str(REPO / "data" / "kaggle_daily"))
    ap.add_argument("--ref-decks", default=str(REPO / "data" / "deck_scan" / "top10_decks.json"))
    ap.add_argument("--out-dir", default=str(REPO / "data" / "il_grimmsnarl_v2"))
    ap.add_argument("--min-overlap", type=int, default=54)
    ap.add_argument("--limit-files-per-day", type=int, default=0, help="0 = no limit (debug only)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-day outputs live in pairs_by_day/ so re-running for one day never
    # clobbers pairs already mined for a different day (each invocation only
    # ever opens ITS OWN day's file in "w" mode). pairs.jsonl / report.json /
    # decklist_stability.json at the top level are the MERGED view across
    # every day file found in pairs_by_day/, rebuilt at the end of each run
    # (so partial/incremental scans stay resumable and cumulative).
    by_day_dir = out_dir / "pairs_by_day"
    by_day_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    report_path = out_dir / "report.json"
    stability_path = out_dir / "decklist_stability.json"

    ref_decks = json.loads(Path(args.ref_decks).read_text())
    targets = set(args.targets)
    for t in targets:
        if t not in ref_decks:
            print(f"[scan] WARNING: no reference deck for target {t!r} in {args.ref_decks}")

    needles = [t.encode("utf-8") for t in targets]

    t0 = time.time()

    for day in args.days:
        day_dir = Path(args.daily_root) / day
        if not day_dir.is_dir():
            print(f"[scan] SKIP missing day dir: {day_dir}")
            continue
        files = sorted(day_dir.glob("*.json"))
        if args.limit_files_per_day:
            files = files[: args.limit_files_per_day]
        print(f"[scan] {day}: {len(files)} JSON files")
        day_stability = {t: [] for t in targets}

        # Each day owns its own pairs/report file under pairs_by_day/, opened
        # fresh ("w") only for THIS day — re-scanning one day never touches
        # (or loses) any other day's already-mined pairs.
        day_pairs_path = by_day_dir / f"{day}.pairs.jsonl"
        day_report_path = by_day_dir / f"{day}.report.json"
        pairs_f = open(day_pairs_path, "w")

        n_files_scanned = 0
        n_grep_hits = 0
        n_episodes_kept = 0
        n_episodes_won = 0
        n_episodes_lost = 0
        n_episodes_filtered_impure = 0
        n_pairs_total = 0
        n_pairs_won = 0
        per_target_stats = {t: {"episodes_kept": 0, "episodes_won": 0, "episodes_impure": 0} for t in targets}

        for fp in files:
            n_files_scanned += 1
            try:
                raw = fp.read_bytes()
            except Exception as e:
                print(f"[scan]   read error {fp.name}: {e}")
                continue

            hit_targets = [t for t, needle in zip(targets, needles) if needle in raw]
            if not hit_targets:
                if n_files_scanned % 1000 == 0:
                    print(f"[scan]   scanned {n_files_scanned}, grep_hits={n_grep_hits}, kept={n_episodes_kept}")
                continue
            n_grep_hits += 1

            try:
                data = json.loads(raw)
            except Exception as e:
                print(f"[scan]   json error {fp.name}: {e}")
                continue

            agents = data.get("info", {}).get("Agents", [])
            rewards = data.get("rewards", [])
            names = [a.get("Name") for a in agents]

            true_decks = true_decks_from_data(data)

            for player_idx, name in enumerate(names):
                if name not in targets:
                    continue
                if name not in hit_targets:
                    # grep matched (e.g. present on other player's side text) but
                    # this specific player slot doesn't carry the name; skip.
                    pass
                won = player_idx < len(rewards) and rewards[player_idx] > 0

                deck = true_decks.get(player_idx)
                overlap = None
                if deck is not None and name in ref_decks:
                    overlap = multiset_overlap(deck, ref_decks[name])
                    day_stability[name].append(overlap)

                if deck is None or name not in ref_decks or overlap < args.min_overlap:
                    n_episodes_filtered_impure += 1
                    per_target_stats[name]["episodes_impure"] += 1
                    continue

                n_episodes_kept += 1
                per_target_stats[name]["episodes_kept"] += 1
                if won:
                    n_episodes_won += 1
                    per_target_stats[name]["episodes_won"] += 1
                else:
                    n_episodes_lost += 1

                # Only keep winner-side decisions (per plan: "obs,action pairs
                # winner-side эксперта") — losing-side play isn't the policy
                # we want to imitate.
                if not won:
                    continue

                episode_pairs = parse_replay(fp)
                for p in episode_pairs:
                    if p["player_index"] != player_idx:
                        continue
                    n_pairs_total += 1
                    n_pairs_won += 1
                    pairs_f.write(json.dumps(p) + "\n")

            if n_files_scanned % 1000 == 0:
                print(f"[scan]   scanned {n_files_scanned}, grep_hits={n_grep_hits}, kept={n_episodes_kept}, pairs={n_pairs_total}")

        pairs_f.close()

        day_stability_summary = {
            t: {
                "n_episodes": len(v),
                "mean_overlap": (sum(v) / len(v) if v else None),
                "min_overlap": (min(v) if v else None),
            }
            for t, v in day_stability.items()
        }
        day_report = {
            "day": day,
            "n_files_scanned": n_files_scanned,
            "n_grep_hits": n_grep_hits,
            "n_episodes_kept": n_episodes_kept,
            "n_episodes_won": n_episodes_won,
            "n_episodes_lost": n_episodes_lost,
            "n_episodes_filtered_impure": n_episodes_filtered_impure,
            "n_pairs_total": n_pairs_total,
            "n_pairs_won": n_pairs_won,
            "per_target_stats": per_target_stats,
            "stability": day_stability_summary,
            "elapsed_seconds": time.time() - t0,
        }
        day_report_path.write_text(json.dumps(day_report, indent=2, ensure_ascii=False))
        print(f"[scan] {day} DONE. files={n_files_scanned} grep_hits={n_grep_hits} kept={n_episodes_kept} "
              f"won={n_episodes_won} lost={n_episodes_lost} impure_filtered={n_episodes_filtered_impure} "
              f"pairs={n_pairs_total}")

    # ── Merge every day found in pairs_by_day/ into the top-level cumulative
    # view (covers days scanned in THIS invocation and any prior ones). ──
    day_report_files = sorted(by_day_dir.glob("*.report.json"))
    merged = {
        "targets": sorted(targets),
        "days_done": [],
        "n_files_scanned": 0,
        "n_grep_hits": 0,
        "n_episodes_kept": 0,
        "n_episodes_won": 0,
        "n_episodes_lost": 0,
        "n_episodes_filtered_impure": 0,
        "n_pairs_total": 0,
        "n_pairs_won": 0,
        "per_target_stats": {t: {"episodes_kept": 0, "episodes_won": 0, "episodes_impure": 0} for t in targets},
    }
    stability_by_day: dict[str, dict] = {}
    with open(pairs_path, "w") as out_f:
        for drp in day_report_files:
            dr = json.loads(drp.read_text())
            merged["days_done"].append(dr["day"])
            for k in ("n_files_scanned", "n_grep_hits", "n_episodes_kept", "n_episodes_won",
                      "n_episodes_lost", "n_episodes_filtered_impure", "n_pairs_total", "n_pairs_won"):
                merged[k] += dr[k]
            for t, stats in dr["per_target_stats"].items():
                if t not in merged["per_target_stats"]:
                    merged["per_target_stats"][t] = {"episodes_kept": 0, "episodes_won": 0, "episodes_impure": 0}
                for k, v in stats.items():
                    merged["per_target_stats"][t][k] += v
            stability_by_day[dr["day"]] = dr["stability"]

            day_pairs_path = by_day_dir / f"{dr['day']}.pairs.jsonl"
            if day_pairs_path.exists():
                with open(day_pairs_path) as in_f:
                    for line in in_f:
                        out_f.write(line)

    merged["days_done"] = sorted(merged["days_done"])
    merged["elapsed_seconds"] = time.time() - t0
    report_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    stability_path.write_text(json.dumps(stability_by_day, indent=2, ensure_ascii=False))

    print(f"[scan] MERGED across {merged['days_done']}. files={merged['n_files_scanned']} "
          f"grep_hits={merged['n_grep_hits']} kept={merged['n_episodes_kept']} "
          f"won={merged['n_episodes_won']} lost={merged['n_episodes_lost']} "
          f"impure_filtered={merged['n_episodes_filtered_impure']} pairs={merged['n_pairs_total']} "
          f"elapsed={merged['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
