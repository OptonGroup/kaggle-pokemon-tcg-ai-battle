#!/usr/bin/env python3
"""subsample_corpus_2b_daily_extension.py — [v082 A] cap the daily-zip extension
(build_corpus_2b_daily_extension.py output) to a manageable EPISODE budget before featurizing.

Why: the qualifying-episode yield from the 4 daily zips turned out far larger than the existing
2B corpus (~12x more decisions from a single-pass scan) — featurize_corpus_2b.py's per-decision
Python feature loop would take hours+ at that scale, well past what this task's time budget
affords, and it's not obviously useful anyway (the marginal value of the 500,000th near-duplicate
grimmsnarl-mirror decision is far below the first 50,000). This script subsamples at the EPISODE
level (never splits a game's decisions across included/excluded — keeps the "no decision-leakage
between held-out state" discipline used everywhere else in this pipeline) down to --target-decisions,
sampled uniformly at random over qualifying episodes (seeded, reproducible) — NOT top-score-only,
to avoid concentrating the extension on a handful of teams' idiosyncratic playstyle.

Usage:
    .venv/bin/python scripts/subsample_corpus_2b_daily_extension.py \\
        --in out/corpus_2b/versions/v082_bc2b/daily_extension \\
        --out out/corpus_2b/versions/v082_bc2b/daily_extension_sampled \\
        --target-decisions 150000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--target-decisions", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_shards = out_dir / "obs_shards"
    out_shards.mkdir(parents=True, exist_ok=True)

    by_episode: dict[int, list[str]] = defaultdict(list)
    with open(in_dir / "decisions_extra.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eid = int(json.loads(line)["episode_id"])
            by_episode[eid].append(line)

    episode_ids = list(by_episode.keys())
    rng = random.Random(args.seed)
    rng.shuffle(episode_ids)

    kept_ids: list[int] = []
    n_decisions = 0
    for eid in episode_ids:
        rows = by_episode[eid]
        if n_decisions and n_decisions + len(rows) > args.target_decisions:
            continue  # keep sampling smaller episodes rather than stopping at the first overshoot
        kept_ids.append(eid)
        n_decisions += len(rows)
        if n_decisions >= args.target_decisions:
            break

    kept_set = set(kept_ids)
    with open(out_dir / "decisions_extra.jsonl", "w") as out_f:
        for eid in kept_ids:
            for line in by_episode[eid]:
                out_f.write(line + "\n")

    n_links = 0
    for eid in kept_ids:
        src = in_dir / "obs_shards" / f"{eid}.jsonl"
        if not src.is_file():
            sys.exit(f"FATAL: missing obs_shard for kept episode {eid}: {src}")
        dest = out_shards / f"{eid}.jsonl"
        dest.symlink_to(src.resolve())
        n_links += 1

    report = {
        "n_qualifying_episodes_available": len(episode_ids),
        "n_qualifying_decisions_available": sum(len(v) for v in by_episode.values()),
        "n_episodes_kept": len(kept_ids),
        "n_decisions_kept": n_decisions,
        "target_decisions": args.target_decisions,
        "seed": args.seed,
    }
    (out_dir / "subsample_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
