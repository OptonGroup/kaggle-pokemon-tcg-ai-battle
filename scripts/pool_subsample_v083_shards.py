#!/usr/bin/env python3
"""pool_subsample_v083_shards.py — pool per-day shard outputs of
build_corpus_2b_daily_extension.py (out/corpus_2b/versions/v083_daily_full/shards/<date>/)
and subsample to a decision budget in ONE streaming pass structure, because the pooled
decisions_extra.jsonl (5.19M rows) does not fit comfortably in 16GB RAM as python strings.

Selection semantics are IDENTICAL to scripts/subsample_corpus_2b_daily_extension.py:
episode-level sampling (never splits a game's decisions), rng.shuffle(episode_ids, seed),
greedy fill that skips episodes overshooting the target rather than stopping.
obs shards are symlinked (one file per episode, names are globally unique episode_ids).

Usage:
    .venv/bin/python scripts/pool_subsample_v083_shards.py \
        --shards out/corpus_2b/versions/v083_daily_full/shards \
        --out out/corpus_2b/versions/v083_daily_full/extension_sampled \
        --target-decisions 600000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-decisions", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shards_root = Path(args.shards)
    out_dir = Path(args.out)
    out_shards = out_dir / "obs_shards"
    out_shards.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(d for d in shards_root.iterdir()
                        if d.is_dir() and (d / "decisions_extra.jsonl").is_file())

    # pass 1: per-episode decision counts + which shard dir the episode lives in
    ep_count: dict[int, int] = {}
    ep_shard: dict[int, Path] = {}
    per_day: dict[str, int] = {}
    for sd in shard_dirs:
        n_day = 0
        with open(sd / "decisions_extra.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # episode_id is the first key build_corpus writes; cheap targeted parse
                eid = int(json.loads(line)["episode_id"])
                if eid in ep_shard and ep_shard[eid] != sd:
                    sys.exit(f"FATAL: episode {eid} appears in two shards: {ep_shard[eid]} and {sd}")
                ep_shard[eid] = sd
                ep_count[eid] = ep_count.get(eid, 0) + 1
                n_day += 1
        per_day[sd.name] = n_day

    episode_ids = list(ep_count.keys())
    rng = random.Random(args.seed)
    rng.shuffle(episode_ids)

    kept: set[int] = set()
    n_decisions = 0
    for eid in episode_ids:
        n = ep_count[eid]
        if n_decisions and n_decisions + n > args.target_decisions:
            continue
        kept.add(eid)
        n_decisions += n
        if n_decisions >= args.target_decisions:
            break

    # pass 2: write kept lines (shard order == chronological; order is irrelevant downstream —
    # featurize/train group by decision_uid and split by episode)
    n_written = 0
    with open(out_dir / "decisions_extra.jsonl", "w") as out_f:
        for sd in shard_dirs:
            with open(sd / "decisions_extra.jsonl") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if int(json.loads(line)["episode_id"]) in kept:
                        out_f.write(line + "\n")
                        n_written += 1
    assert n_written == n_decisions, (n_written, n_decisions)

    n_links = 0
    for eid in kept:
        src = ep_shard[eid] / "obs_shards" / f"{eid}.jsonl"
        if not src.is_file():
            sys.exit(f"FATAL: missing obs_shard for kept episode {eid}: {src}")
        dest = out_shards / f"{eid}.jsonl"
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src.resolve())
        n_links += 1

    kept_per_day = {d: 0 for d in per_day}
    for eid in kept:
        kept_per_day[ep_shard[eid].name] += 1

    report = {
        "n_shard_dirs": len(shard_dirs),
        "n_qualifying_episodes_available": len(episode_ids),
        "n_qualifying_decisions_available": sum(ep_count.values()),
        "n_episodes_kept": len(kept),
        "n_decisions_kept": n_decisions,
        "target_decisions": args.target_decisions,
        "seed": args.seed,
        "available_decisions_per_day": per_day,
        "kept_episodes_per_day": kept_per_day,
    }
    (out_dir / "subsample_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
