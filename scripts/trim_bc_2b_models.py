#!/usr/bin/env python3
"""trim_bc_2b_models.py — pre-trim LightGBM group models (keep first N trees) BEFORE
export_bc_2b.py, so the export's built-in size-pruning path (the v082 trap: it drops
trees by its own heuristic and degrades top1) never triggers.

Truncation = Booster.save_model(num_iteration=N) — keeps the FIRST N boosting rounds,
i.e. the same prefix the early-stopped validation optimum was built from.

Usage:
    .venv/bin/python scripts/trim_bc_2b_models.py \
        --in out/corpus_2b/versions/v083_daily_full/models_seed42 \
        --out out/corpus_2b/versions/v083_daily_full/models_seed42_trimmed \
        --keep main=440 c7=135 low=65 mid=66 easy=22 rest=1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--keep", nargs="+", required=True, help="group=N pairs; N=0 -> copy untrimmed")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {}
    for pair in args.keep:
        g, n = pair.split("=")
        keep[g] = int(n)

    for g, n in keep.items():
        src = in_dir / f"{g}.txt"
        booster = lgb.Booster(model_file=str(src))
        total = booster.num_trees()
        n_keep = min(n, total) if n > 0 else total
        booster.save_model(str(out_dir / f"{g}.txt"), num_iteration=n_keep)
        check = lgb.Booster(model_file=str(out_dir / f"{g}.txt"))
        print(f"{g}: {total} -> {check.num_trees()} trees "
              f"({src.stat().st_size} -> {(out_dir / f'{g}.txt').stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
