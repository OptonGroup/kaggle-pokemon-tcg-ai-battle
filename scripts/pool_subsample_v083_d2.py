#!/usr/bin/env python3
"""pool_subsample_v083_d2.py — D2 variant of pool_subsample_v083_shards.py: SAME 600k
decision budget, different sampling POLICY (the only lever changed vs v083/D1):

  weight(episode) = w_mirror * w_recency
    w_mirror  = 3.0  if opp is mirror AND donor mirror-WR >= 0.55 (n_mirror >= 30)
              = 0.5  if opp is mirror AND (donor mirror-WR < 0.50 OR n_mirror < 30)
              = 1.0  otherwise (non-mirror episodes, and mirror from 0.50..0.55 donors)
    w_recency = 0.5 ** (age_days / 21)   (age vs newest shard day)

Weighted sampling WITHOUT replacement via Efraimidis-Spirakis A-Res keys
(key = u^(1/w)), then the same greedy fill (skip overshoot) as D1/v082.
Donor mirror-WR is computed from the FULL pool (episode-level, all shards).

Usage:
    .venv/bin/python scripts/pool_subsample_v083_d2.py \
        --shards out/corpus_2b/versions/v083_daily_full/shards \
        --out out/corpus_2b/versions/v083_daily_full/extension_sampled_d2 \
        --target-decisions 600000 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-decisions", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--half-life-days", type=float, default=21.0)
    ap.add_argument("--mirror-strong-wr", type=float, default=0.55)
    ap.add_argument("--mirror-weak-wr", type=float, default=0.50)
    ap.add_argument("--mirror-min-n", type=int, default=30)
    ap.add_argument("--w-strong", type=float, default=3.0)
    ap.add_argument("--w-weak", type=float, default=0.5)
    args = ap.parse_args()

    shards_root = Path(args.shards)
    out_dir = Path(args.out)
    out_shards = out_dir / "obs_shards"
    out_shards.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(d for d in shards_root.iterdir()
                        if d.is_dir() and (d / "decisions_extra.jsonl").is_file())
    newest = max(dt.date.fromisoformat(d.name) for d in shard_dirs)

    # pass 1: per-episode-side meta + donor mirror stats over the FULL pool
    ep_count: dict[int, int] = {}
    ep_shard: dict[int, Path] = {}
    ep_meta: dict[int, tuple[str, str, str]] = {}  # eid -> (expert, opp_archetype, day)
    donor_mirror = defaultdict(lambda: [0, 0])  # expert -> [n_mirror_eps, wins]
    seen_sides: set[tuple[int, int]] = set()
    for sd in shard_dirs:
        with open(sd / "decisions_extra.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                eid = int(d["episode_id"])
                if eid in ep_shard and ep_shard[eid] != sd:
                    sys.exit(f"FATAL: episode {eid} in two shards")
                ep_shard[eid] = sd
                ep_count[eid] = ep_count.get(eid, 0) + 1
                ep_meta[eid] = (d["expert"], d["opp_archetype"], sd.name)
                side = (eid, int(d["agent_index"]))
                if side not in seen_sides:
                    seen_sides.add(side)
                    if d["opp_archetype"] == "mirror_grimmsnarl":
                        st = donor_mirror[d["expert"]]
                        st[0] += 1
                        st[1] += 1 if d["game_won"] else 0

    def w_mirror_for(expert: str, opp: str) -> float:
        if opp != "mirror_grimmsnarl":
            return 1.0
        n, w = donor_mirror.get(expert, [0, 0])
        if n >= args.mirror_min_n and w / n >= args.mirror_strong_wr:
            return args.w_strong
        if n < args.mirror_min_n or w / n < args.mirror_weak_wr:
            return args.w_weak
        return 1.0

    rng = random.Random(args.seed)
    keyed = []
    for eid, (expert, opp, day) in ep_meta.items():
        age = (newest - dt.date.fromisoformat(day)).days
        w = w_mirror_for(expert, opp) * (0.5 ** (age / args.half_life_days))
        u = rng.random()
        keyed.append((u ** (1.0 / w), eid))
    keyed.sort(reverse=True)

    kept: set[int] = set()
    n_decisions = 0
    for _, eid in keyed:
        n = ep_count[eid]
        if n_decisions and n_decisions + n > args.target_decisions:
            continue
        kept.add(eid)
        n_decisions += n
        if n_decisions >= args.target_decisions:
            break

    n_written = 0
    with open(out_dir / "decisions_extra.jsonl", "w") as out_f:
        for sd in shard_dirs:
            with open(sd / "decisions_extra.jsonl") as f:
                for line in f:
                    line = line.strip()
                    if line and int(json.loads(line)["episode_id"]) in kept:
                        out_f.write(line + "\n")
                        n_written += 1
    assert n_written == n_decisions, (n_written, n_decisions)

    for eid in kept:
        src = ep_shard[eid] / "obs_shards" / f"{eid}.jsonl"
        if not src.is_file():
            sys.exit(f"FATAL: missing obs_shard {src}")
        dest = out_shards / f"{eid}.jsonl"
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src.resolve())

    kept_mirror = sum(1 for e in kept if ep_meta[e][1] == "mirror_grimmsnarl")
    kept_strong = sum(1 for e in kept if ep_meta[e][1] == "mirror_grimmsnarl"
                      and w_mirror_for(ep_meta[e][0], "mirror_grimmsnarl") == args.w_strong)
    kept_per_day: dict[str, int] = defaultdict(int)
    for e in kept:
        kept_per_day[ep_meta[e][2]] += 1

    report = {
        "policy": {"half_life_days": args.half_life_days,
                   "mirror_strong_wr": args.mirror_strong_wr, "mirror_weak_wr": args.mirror_weak_wr,
                   "mirror_min_n": args.mirror_min_n, "w_strong": args.w_strong, "w_weak": args.w_weak},
        "n_qualifying_episodes_available": len(ep_meta),
        "n_decisions_kept": n_decisions,
        "n_episodes_kept": len(kept),
        "kept_mirror_episodes": kept_mirror,
        "kept_mirror_strong_donor_episodes": kept_strong,
        "target_decisions": args.target_decisions,
        "seed": args.seed,
        "kept_episodes_per_day": dict(sorted(kept_per_day.items())),
    }
    (out_dir / "subsample_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    print("kept_per_day (последние 10):", dict(list(sorted(kept_per_day.items()))[-10:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
