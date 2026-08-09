"""Разовый гейт (2026-07-23): закрыть дыру в оценке равновесного рейтинга v070 —
прямой замер WR v070_grimmsnarl_hybrid против Great Tusk / Crustle library-out
(единственный внешний архетип-контр к ex, что есть локально; Starmie/Dragapult
проксей у нас нет). Переиспользует cg_native run_pair из gate_v064_regression.

Кандидат: v070 из worktree (submissions/v070_grimmsnarl_hybrid, пилот hybrid + наша дека).
Оппонент: data/external_agents/tusk_crustle_libout/tusk_crustle_submission.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_v064_regression as g  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_DIR = os.path.join(
    _REPO, ".claude/worktrees/v070-grimmsnarl-hybrid/submissions/v070_grimmsnarl_hybrid"
)
TUSK_DIR = os.path.join(
    _REPO, "data/external_agents/tusk_crustle_libout/tusk_crustle_submission"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default=os.path.join(g.GATES_DIR, "v070_vs_tusk.json"))
    args = ap.parse_args()

    assert os.path.isdir(CANDIDATE_DIR), f"нет v070: {CANDIDATE_DIR}"
    assert os.path.isdir(TUSK_DIR), f"нет tusk: {TUSK_DIR}"

    print(f"[run] v070 vs tusk_crustle_libout: n={args.n} ...", flush=True)
    stats = g.run_pair(CANDIDATE_DIR, "sub:" + TUSK_DIR, "tusk_crustle",
                       args.n, args.seed, args.workers)
    print(f"[done] tusk_crustle: WR={stats['win_rate']:.4f} "
          f"[{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}] "
          f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}", flush=True)

    out = dict(tag="v070_vs_tusk", candidate="v070_grimmsnarl_hybrid",
               opponent="tusk_crustle_libout", opponents={"tusk_crustle": stats})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
