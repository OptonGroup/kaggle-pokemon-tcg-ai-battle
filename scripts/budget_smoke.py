#!/usr/bin/env python3
"""budget_smoke.py — prove the v030 time budget actually drives the search.

Runs one self-play game with the candidate's agent() on both sides and reports,
per decision: wall time, simulations run, and whether MCTS or the fallback
produced the move. Mimics the Kaggle loader (exec without __file__) because the
v026-v028 failures were all silent fallbacks that a plain import never caught.

    python3 scripts/budget_smoke.py --version v030_time_budget --overage 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def load_agent(version: str, overage: float):
    """Load main.py the way Kaggle does: exec into a bare namespace, no __file__."""
    sub_dir = REPO / "submissions" / version
    sys.path.insert(0, str(sub_dir))
    src = (sub_dir / "main.py").read_text()
    ns: dict = {"__name__": "__main__"}
    exec(compile(src, "main.py", "exec"), ns)
    return ns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v030_time_budget")
    p.add_argument("--overage", type=float, default=600.0,
                   help="remainingOverageTime to advertise to the agent")
    p.add_argument("--max-moves", type=int, default=40)
    args = p.parse_args()

    from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

    print(f"[smoke] loading {args.version} (bare namespace, no __file__)", flush=True)
    ns = load_agent(args.version, args.overage)
    agent = ns["agent"]

    deck = agent({"select": None})
    print(f"[smoke] deck: {len(deck)} cards", flush=True)

    obs_dict, _start = battle_start(list(deck), list(deck))
    remaining = args.overage
    move_times = []

    for move in range(args.max_moves):
        sel = obs_dict.get("select")
        if sel is None:
            break
        obs_dict["remainingOverageTime"] = remaining
        t0 = time.perf_counter()
        choice = agent(obs_dict)
        dt = time.perf_counter() - t0
        remaining -= dt
        move_times.append(dt)
        mcts = ns.get("_n_mcts_decisions", 0)
        fb = ns.get("_n_fallback_decisions", 0)
        exc = ns.get("_n_exceptions", 0)
        print(f"[smoke] move={move:3d} time={dt:6.3f}s remaining={remaining:7.1f}s "
              f"mcts={mcts} fallback={fb} exc={exc} choice={choice}", flush=True)
        if remaining <= 0:
            print("[smoke] BANK EXHAUSTED — would be DeadlineExceeded on Kaggle", flush=True)
            break
        obs_dict = battle_select(choice)
        if obs_dict.get("current", {}).get("result", -1) >= 0:
            print("[smoke] game finished", flush=True)
            break

    battle_finish()

    mcts = ns.get("_n_mcts_decisions", 0)
    fb = ns.get("_n_fallback_decisions", 0)
    exc = ns.get("_n_exceptions", 0)
    spent = args.overage - remaining
    print("\n=== SUMMARY ===")
    print(f"moves        : {len(move_times)}")
    print(f"mcts / fallback / exceptions : {mcts} / {fb} / {exc}")
    print(f"time spent   : {spent:.1f}s of {args.overage:.0f}s bank ({100*spent/args.overage:.1f}%)")
    if move_times:
        print(f"per-move     : median {sorted(move_times)[len(move_times)//2]:.3f}s  "
              f"max {max(move_times):.3f}s")
    verdict = "PASS" if mcts > 0 and exc == 0 else "FAIL"
    print(f"verdict      : {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
