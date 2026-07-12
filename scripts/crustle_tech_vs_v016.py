"""crustle_tech_vs_v016.py — head-to-head: Crustle candidate (baseline/tech) vs v016_alakazam
(REAL IL pilot, not greedy). Deck-tech audit for the Alakazam hole (PLAN.md / deck-tech task 07-12).

Both sides use their own trained IL pilot + their own deck.csv. Seat-swap, multiprocessing
spawn, resumable state (matches h3_gate.py / field_viability.py conventions).

Запуск:
    .venv/bin/python scripts/crustle_tech_vs_v016.py --games 150 --workers 4 \
        --candidates v009a_crustle,v020_crustle_techA,v021_crustle_techB,v022_crustle_techC
    .venv/bin/python scripts/crustle_tech_vs_v016.py --report-only
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
OPPONENT = "v016_alakazam"

RESULTS_DIR = os.path.join(ROOT, "results", "gates")
STATE_PATH = os.path.join(RESULTS_DIR, "tech_vs_v016_state.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "tech_vs_v016_report.json")

DEFAULT_CANDIDATES = [
    "v009a_crustle",
    "v020_crustle_techA",
    "v021_crustle_techB",
    "v022_crustle_techC",
]


def play_one(task: dict) -> dict:
    cand = task["candidate"]
    our_first = task["our_first"]

    our_mod = _load_submission_module(cand)
    our_agent = our_mod.agent
    our_deck = list(getattr(our_mod, "my_deck", None) or _resolve_opp_deck(cand))

    opp_mod = _load_submission_module(OPPONENT)
    opp_agent = opp_mod.agent
    opp_deck = list(getattr(opp_mod, "my_deck", None) or _resolve_opp_deck(OPPONENT))

    if our_first:
        deck0, deck1 = our_deck, opp_deck
        agent0, agent1 = our_agent, opp_agent
    else:
        deck0, deck1 = opp_deck, our_deck
        agent0, agent1 = opp_agent, our_agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(candidate=cand, our_first=our_first, outcome="crash",
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

    return dict(candidate=cand, our_first=our_first, outcome=outcome, winner=winner,
                moves=move_count, n_illegal=n_illegal, n_exceptions=n_exceptions, exc_side=exc_side)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cells": {}}


def save_state(state: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def build_queue(state: dict, candidates: list[str], n_games: int) -> list[dict]:
    cells = state.get("cells", {})
    tasks: list[dict] = []
    for gi in range(n_games):
        for cand in candidates:
            have = len(cells.get(cand, {}).get("games", []))
            if have > gi:
                continue
            tasks.append(dict(candidate=cand, game_idx=gi, our_first=(gi % 2 == 0)))
    return tasks


def record(state: dict, res: dict) -> None:
    cells = state.setdefault("cells", {})
    ck = res["candidate"]
    if ck not in cells:
        cells[ck] = {"candidate": ck, "games": []}
    cells[ck]["games"].append(res)


def build_report(state: dict, candidates: list[str]) -> dict:
    cells = state.get("cells", {})
    rows = {}
    for cand in candidates:
        games = cells.get(cand, {}).get("games", [])
        w = sum(1 for g in games if g["outcome"] == "our_win")
        l = sum(1 for g in games if g["outcome"] == "opp_win")
        d = sum(1 for g in games if g["outcome"] == "draw")
        bad = sum(1 for g in games if g["outcome"] in ("crash", "illegal", "exception", "aborted"))
        n = len(games)
        dec = w + l
        wr = (w / dec) if dec else float("nan")
        lo, hi = wilson_ci(w, dec) if dec else (float("nan"), float("nan"))
        rows[cand] = dict(n=n, w=w, l=l, d=d, bad=bad, wr=wr, wilson_lo=lo, wilson_hi=hi)
    return dict(opponent=OPPONENT, candidates=candidates, rows=rows)


def print_report(rep: dict) -> None:
    print("\n" + "=" * 80)
    print(f"CRUSTLE TECH vs {rep['opponent']} (REAL IL pilot, seat-swap)")
    print("=" * 80)
    for cand, r in rep["rows"].items():
        wr = r["wr"]
        wr_s = f"{wr:.3f}" if not math.isnan(wr) else "nan"
        print(f"  {cand:28s} {r['w']:3d}-{r['l']:3d} (d={r['d']}, bad={r['bad']}) "
              f"WR={wr_s} Wilson[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] n={r['n']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    state = load_state()

    if args.report_only:
        rep = build_report(state, candidates)
        print_report(rep)
        return 0

    queue = build_queue(state, candidates, args.games)
    print(f"Кандидаты: {candidates}  vs {OPPONENT}")
    print(f"Партий/candidate: {args.games}  →  осталось сыграть: {len(queue)} игр")
    if not queue:
        print("Всё уже сыграно.")
        rep = build_report(state, candidates)
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
                    res = dict(candidate=task["candidate"], our_first=task["our_first"],
                               outcome="crash", error=repr(e), n_illegal=0, n_exceptions=1, moves=0)
                record(state, res)
                n_done += 1
                if n_done % save_every == 0:
                    save_state(state)
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

    save_state(state)
    rep = build_report(state, candidates)
    with open(REPORT_PATH, "w") as f:
        json.dump(rep, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print_report(rep)
    print(f"\nState: {STATE_PATH}\nReport: {REPORT_PATH}")
    print(f"Wall: {(time.time()-t_start)/60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
