"""gate_v034_romanrozen.py — head-to-head gate: v034_romanrozen_clone (Archaludon ex /
Cinderace metal-tempo clone of the public romanrozen/strong-start-baseline-agent-v10-lb-950
notebook) vs a configurable opponent (default v009a_crustle, our historical live champion).

Same pattern as scripts/crustle_tech_vs_v016.py: multiprocessing spawn, resumable JSON state,
seat-swap, cg_native engine (macOS ARM, no Docker). Reuses helpers from h3_gate.py
(_load_submission_module, _resolve_opp_deck, _is_legal, _median, wilson_ci) unchanged.

Запуск:
    .venv/bin/python scripts/gate_v034_romanrozen.py --games 4 --workers 2 --opponent v009a_crustle
    .venv/bin/python scripts/gate_v034_romanrozen.py --games 100 --workers 4 --opponent v009a_crustle
    .venv/bin/python scripts/gate_v034_romanrozen.py --games 30 --workers 4 --opponent v001_random
    .venv/bin/python scripts/gate_v034_romanrozen.py --report-only --opponent v009a_crustle
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

import h3_gate  # noqa: E402
from h3_gate import (  # noqa: E402
    _load_submission_module,
    _resolve_opp_deck,
    _is_legal,
    _median,
    wilson_ci,
)

MAX_MOVES_PER_GAME = 2000

RESULTS_DIR = os.path.join(ROOT, "results", "gates")

DEFAULT_CANDIDATES = [
    "v034_romanrozen_clone",
]


def _state_path(opponent: str) -> str:
    return os.path.join(RESULTS_DIR, f"v034_romanrozen_vs_{opponent}_state.json")


def _report_path(opponent: str) -> str:
    return os.path.join(RESULTS_DIR, f"v034_romanrozen_vs_{opponent}_report.json")


def play_one(task: dict) -> dict:
    cand = task["candidate"]
    opponent = task["opponent"]
    our_first = task["our_first"]

    our_mod = _load_submission_module(cand)
    our_agent = our_mod.agent
    our_deck = list(getattr(our_mod, "my_deck", None) or _resolve_opp_deck(cand))

    opp_mod = _load_submission_module(opponent)
    opp_agent = opp_mod.agent
    opp_deck = list(getattr(opp_mod, "my_deck", None) or _resolve_opp_deck(opponent))

    if our_first:
        deck0, deck1 = our_deck, opp_deck
        agent0, agent1 = our_agent, opp_agent
    else:
        deck0, deck1 = opp_deck, our_deck
        agent0, agent1 = opp_agent, our_agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(candidate=cand, opponent=opponent, our_first=our_first, outcome="crash",
                    error=f"battle_start:{getattr(start,'errorType',None)}",
                    n_illegal=0, n_exceptions=1, moves=0)

    move_count = 0
    n_illegal = n_exceptions = 0
    winner = None
    exc_side = None
    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            sel = obs_dict.get("select")
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                is_our_turn = (your_index == 0) == our_first
                exc_side = "our" if is_our_turn else "opp"
                break
            if not _is_legal(choice, sel):
                n_illegal += 1
                break
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    our_side = 0 if our_first else 1
    if n_exceptions > 0:
        outcome = "exception"
    elif n_illegal > 0:
        outcome = "illegal"
    elif winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return dict(candidate=cand, opponent=opponent, our_first=our_first, outcome=outcome, winner=winner,
                moves=move_count, n_illegal=n_illegal, n_exceptions=n_exceptions, exc_side=exc_side)


def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"cells": {}}


def save_state(state: dict, state_path: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


def build_queue(state: dict, candidates: list[str], opponent: str, n_games: int) -> list[dict]:
    cells = state.get("cells", {})
    tasks: list[dict] = []
    for gi in range(n_games):
        for cand in candidates:
            have = len(cells.get(cand, {}).get("games", []))
            if have > gi:
                continue
            tasks.append(dict(candidate=cand, opponent=opponent, game_idx=gi, our_first=(gi % 2 == 0)))
    return tasks


def record(state: dict, res: dict) -> None:
    cells = state.setdefault("cells", {})
    ck = res["candidate"]
    if ck not in cells:
        cells[ck] = {"candidate": ck, "games": []}
    cells[ck]["games"].append(res)


def build_report(state: dict, candidates: list[str], opponent: str) -> dict:
    cells = state.get("cells", {})
    rows = {}
    for cand in candidates:
        games = cells.get(cand, {}).get("games", [])
        w = sum(1 for g in games if g["outcome"] == "our_win")
        l = sum(1 for g in games if g["outcome"] == "opp_win")
        d = sum(1 for g in games if g["outcome"] == "draw")
        bad = sum(1 for g in games if g["outcome"] in ("crash", "illegal", "exception", "aborted"))
        n_exceptions = sum(g.get("n_exceptions", 0) for g in games)
        n_illegal = sum(g.get("n_illegal", 0) for g in games)
        n_aborted = sum(1 for g in games if g["outcome"] == "aborted")
        moves = [g["moves"] for g in games if g.get("moves")]
        n = len(games)
        dec = w + l
        wr = (w / dec) if dec else float("nan")
        lo, hi = wilson_ci(w, dec) if dec else (float("nan"), float("nan"))
        rows[cand] = dict(n=n, w=w, l=l, d=d, bad=bad, wr=wr, wilson_lo=lo, wilson_hi=hi,
                           n_exceptions=n_exceptions, n_illegal=n_illegal, n_aborted=n_aborted,
                           median_moves=_median(moves), mean_moves=(sum(moves) / len(moves) if moves else 0.0))
    return dict(opponent=opponent, candidates=candidates, rows=rows)


def print_report(rep: dict) -> None:
    print("\n" + "=" * 80)
    print(f"v034_romanrozen_clone vs {rep['opponent']} (seat-swap)")
    print("=" * 80)
    for cand, r in rep["rows"].items():
        wr = r["wr"]
        wr_s = f"{wr:.3f}" if not math.isnan(wr) else "nan"
        print(f"  {cand:28s} {r['w']:3d}-{r['l']:3d} (d={r['d']}, bad={r['bad']}) "
              f"WR={wr_s} Wilson[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] n={r['n']}  "
              f"exc={r['n_exceptions']} illegal={r['n_illegal']} aborted={r['n_aborted']} "
              f"median_moves={r['median_moves']:.0f} mean_moves={r['mean_moves']:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opponent", type=str, default="v009a_crustle")
    ap.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    opponent = args.opponent
    state_path = _state_path(opponent)
    report_path = _report_path(opponent)
    state = load_state(state_path)

    if args.report_only:
        rep = build_report(state, candidates, opponent)
        print_report(rep)
        return 0

    queue = build_queue(state, candidates, opponent, args.games)
    print(f"Кандидаты: {candidates}  vs {opponent}")
    print(f"Партий/candidate: {args.games}  →  осталось сыграть: {len(queue)} игр")
    if not queue:
        print("Всё уже сыграно.")
        rep = build_report(state, candidates, opponent)
        print_report(rep)
        return 0

    ctx = __import__("multiprocessing").get_context("spawn")
    t_start = time.time()
    n_done = 0
    save_every = max(4, len(queue) // 50)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {}
        it = iter(queue)
        for _ in range(min(args.workers * 2, len(queue))):
            try:
                task = next(it)
            except StopIteration:
                break
            futures[ex.submit(play_one, task)] = task

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                task = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(candidate=task["candidate"], opponent=task["opponent"],
                               our_first=task["our_first"],
                               outcome="crash", error=repr(e), n_illegal=0, n_exceptions=1, moves=0)
                record(state, res)
                n_done += 1
                if n_done % save_every == 0:
                    save_state(state, state_path)
                    el = time.time() - t_start
                    rate = n_done / el if el else 0
                    eta = (len(queue) - n_done) / rate if rate else 0
                    print(f"  [{n_done}/{len(queue)}] {rate:.2f} игр/с  ETA {eta/60:.0f}мин  "
                          f"last: {res['candidate'][:20]} -> {res['outcome']}", flush=True)
                try:
                    nt = next(it)
                    futures[ex.submit(play_one, nt)] = nt
                except StopIteration:
                    pass

    save_state(state, state_path)
    rep = build_report(state, candidates, opponent)
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print_report(rep)
    print(f"\nState: {state_path}\nReport: {report_path}")
    print(f"Wall: {(time.time()-t_start)/60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
