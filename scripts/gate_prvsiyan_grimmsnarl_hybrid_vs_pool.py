"""Разведочный гейт (2026-07-23, "ради прикола"): прогоняет внешний агент
prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1 (submissions/external_prvsiyan_grimmsnarl_hybrid)
как КАНДИДАТА против того же CORE_PAIRS пула, что и v064 (gate_v064_regression.py),
плюс явный h2h против v064 самой. Переиспользует run_pair/resolve_dir/CORE_PAIRS
оттуда, ничего не трогает в v064-скрипте.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_v064_regression as g  # noqa: E402

CANDIDATE_DIR = g.resolve_dir("submissions/external_prvsiyan_grimmsnarl_hybrid")

PAIRS = list(g.CORE_PAIRS) + [
    dict(key="v064_h2h", opp_dir=g.resolve_dir("submissions/v064_starmie_scaling"), baseline=None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=os.path.join(g.GATES_DIR, "ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json"))
    args = ap.parse_args()

    existing = g.load_existing(args.out)
    opponents = existing.get("opponents", {})
    for cfg in PAIRS:
        key = cfg["key"]
        if args.only and key != args.only:
            continue
        have = opponents.get(key, {})
        if have.get("n_games", 0) >= args.n:
            print(f"[skip] {key}: already have n_games={have['n_games']}")
            continue
        print(f"[run] {key}: n={args.n} vs {cfg['opp_dir']} ...", flush=True)
        stats = g.run_pair(CANDIDATE_DIR, "sub:" + cfg["opp_dir"], key, args.n, args.seed, args.workers)
        print(f"[done] {key}: WR={stats['win_rate']:.4f} "
              f"[{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}] "
              f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}", flush=True)
        opponents[key] = stats
        out = dict(
            tag="ext_prvsiyan_grimmsnarl_hybrid_vs_pool",
            agent=dict(agent_dir="external_prvsiyan_grimmsnarl_hybrid",
                       label="external:prvsiyan_grimmsnarl_hybrid"),
            opponents=opponents,
        )
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
