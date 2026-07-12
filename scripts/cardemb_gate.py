#!/usr/bin/env python3
"""cardemb_gate.py — local gate for the card-embedding BC agent (scripts/cardemb_agent.py).

PLAN_GOLD "moment of truth": does the card-embedding model (58.3% val top-1)
play STRONGER than our LightGBM Crustle champion (v009a) when both play the
IDENTICAL Crustle deck? This isolates the "model" variable from "deck" choice.

Three modes (reuses scripts/h3_gate.py's battle-tested loaders/greedy-pilot/
cg-alias/wilson_ci — does NOT modify h3_gate.py, only imports from it):

  sanity  — 1-2 games, cardemb agent vs random_agent. Checks 0 illegal/0
            exceptions and that the agent doesn't just spam END.
  h2h     — cardemb agent vs v009a (LightGBM) on the SAME Crustle deck,
            seat-swapped. THE headline number: WR of cardemb vs v009a.
  field   — cardemb agent vs 19 meta decks via the generic greedy pilot
            (same protocol as scripts/field_viability.py), for comparison
            against v009a's already-measured field WR.

Multiprocessing spawn (native macOS arm64 engine; threading is unsafe, see
CLAUDE.md). Each worker imports cardemb_agent fresh and lazily loads the torch
model on first call — spawn-safe, no attempt to pickle the model itself.

Usage:
  .venv/bin/python scripts/cardemb_gate.py --mode sanity --games 1
  .venv/bin/python scripts/cardemb_gate.py --mode h2h --games 100 --workers 8
  .venv/bin/python scripts/cardemb_gate.py --mode field --games 10 --workers 8
  .venv/bin/python scripts/cardemb_gate.py --mode h2h --report-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# Defense-in-depth: same macOS libomp double-init guard as cardemb_agent.py.
# Set here too (applies in every spawned worker, since spawn re-imports this
# module) in case a worker ever loads lightgbm (v009a) before cardemb_agent.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (SRC, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

import h3_gate  # noqa: E402  — sets up sys.modules cg-alias as an import side-effect
from h3_gate import (  # noqa: E402
    greedy_meta_agent,
    _load_submission_module,
    _load_deck_csv,
    _resolve_opp_deck,
    _load_meta_deck,
    _is_legal,
    _median,
    wilson_ci,
)

MAX_MOVES_PER_GAME = 2000
RESULTS_DIR = os.path.join(ROOT, "results", "gates")
META_DIR = os.path.join(ROOT, "decks", "meta")


def _list_meta_decks() -> list[str]:
    out = []
    for fn in sorted(os.listdir(META_DIR)):
        if fn.endswith(".csv"):
            out.append(fn[:-4])
    return out


def _random_agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        # random_agent still needs a legal 60-card deck; reuse v001's deck.
        return _resolve_opp_deck("v001_random")
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    max_c = int(sel.get("maxCount") or 1)
    min_c = int(sel.get("minCount") or 1)
    k = random.randint(min_c, max_c) if max_c > min_c else max_c
    k = min(k, n)
    return random.sample(range(n), k)


# ---------------------------------------------------------------------------
# Per-game worker (all heavy objects constructed INSIDE — spawn-safe)
# ---------------------------------------------------------------------------

def _make_player(name: str):
    """Returns (agent_fn, deck) for a named player: 'cardemb', 'v009a_crustle', 'random', or a submissions/ dir name."""
    if name == "cardemb":
        import cardemb_agent
        return cardemb_agent.agent, list(cardemb_agent.CRUSTLE_DECK)
    if name == "random":
        return _random_agent, _resolve_opp_deck("v001_random")
    mod = _load_submission_module(name)
    deck = list(getattr(mod, "my_deck", None) or _resolve_opp_deck(name))
    return mod.agent, deck


def play_one(task: dict) -> dict:
    p0_name = task["p0"]
    p1_name = task["p1"]
    opp_is_meta = task.get("meta_deck")  # deck name if opponent is a meta greedy pilot, else None
    our_first = task["our_first"]  # which side is "ours" (cardemb) for outcome bookkeeping
    seed = task["seed"]

    random.seed(seed)

    if opp_is_meta:
        # one side is cardemb, the other is greedy_meta_agent on a named meta deck
        cardemb_agent_fn, cardemb_deck = _make_player("cardemb")
        opp_deck = _load_meta_deck(opp_is_meta)
        opp_agent_fn = greedy_meta_agent
        if our_first:
            deck0, deck1 = cardemb_deck, opp_deck
            agent0, agent1 = cardemb_agent_fn, opp_agent_fn
        else:
            deck0, deck1 = opp_deck, cardemb_deck
            agent0, agent1 = opp_agent_fn, cardemb_agent_fn
    else:
        agent0_fn, deck0_ = _make_player(p0_name)
        agent1_fn, deck1_ = _make_player(p1_name)
        deck0, deck1 = deck0_, deck1_
        agent0, agent1 = agent0_fn, agent1_fn

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(**task, outcome="crash", error=f"battle_start:{getattr(start, 'errorType', None)}",
                    n_illegal=0, n_exceptions=1, moves=0, ms_median=0.0, ms_max=0.0,
                    n_model_decisions=None, n_fallback_decisions=None, n_agent_exceptions=None)

    move_count = 0
    n_illegal = n_exceptions = 0
    winner = None
    our_times: list[float] = []
    exc_side = None
    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            is_our_turn = (your_index == 0) == our_first
            agent_fn = agent0 if your_index == 0 else agent1
            sel = obs_dict.get("select")
            t0 = time.perf_counter()
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                break
            if is_our_turn:
                our_times.append((time.perf_counter() - t0) * 1000.0)
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

    # Pull cardemb instrumentation counters if cardemb was involved (best-effort;
    # module-level globals only reflect THIS process's calls, which is fine since
    # each worker process plays whole games sequentially).
    n_model_dec = n_fallback_dec = n_agent_exc = None
    try:
        import cardemb_agent
        n_model_dec = cardemb_agent._n_model_decisions
        n_fallback_dec = cardemb_agent._n_fallback_decisions
        n_agent_exc = cardemb_agent._n_exceptions
    except Exception:
        pass

    return dict(**task, outcome=outcome, exc_side=exc_side, winner=winner, moves=move_count,
                n_illegal=n_illegal, n_exceptions=n_exceptions,
                ms_median=_median(our_times), ms_max=(max(our_times) if our_times else 0.0),
                n_model_decisions=n_model_dec, n_fallback_decisions=n_fallback_dec,
                n_agent_exceptions=n_agent_exc)


# ---------------------------------------------------------------------------
# State (resumable, one JSON file per mode)
# ---------------------------------------------------------------------------

def state_path(mode: str) -> str:
    return os.path.join(RESULTS_DIR, f"cardemb_gate_{mode}_state.json")


def report_path(mode: str) -> str:
    return os.path.join(RESULTS_DIR, f"cardemb_gate_{mode}_report.json")


def load_state(mode: str) -> dict:
    p = state_path(mode)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"games": []}


def save_state(mode: str, state: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = state_path(mode) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path(mode))


def cell_key(res: dict) -> str:
    return res.get("meta_deck") or f"{res['p0']}_vs_{res['p1']}"


# ---------------------------------------------------------------------------
# Task builders per mode
# ---------------------------------------------------------------------------

def build_tasks_sanity(n_games: int, have: int) -> list[dict]:
    return [dict(p0="cardemb", p1="random", meta_deck=None, our_first=(i % 2 == 0),
                 seed=20260713 + i, game_idx=i)
            for i in range(have, n_games)]


def build_tasks_h2h(n_games: int, have: int) -> list[dict]:
    return [dict(p0="cardemb", p1="v009a_crustle", meta_deck=None, our_first=(i % 2 == 0),
                 seed=20260713 + i, game_idx=i)
            for i in range(have, n_games)]


def build_tasks_field(n_games: int, state: dict, decks: list[str]) -> list[dict]:
    cells = state.get("cells", {})
    tasks = []
    for gi in range(n_games):
        for deck in decks:
            have = len(cells.get(deck, []))
            if have > gi:
                continue
            tasks.append(dict(p0="cardemb", p1=None, meta_deck=deck, our_first=(gi % 2 == 0),
                               seed=20260713 + hash((deck, gi)) % 1_000_000, game_idx=gi))
    return tasks


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def cell_stats(games: list[dict]) -> dict:
    wins = sum(1 for g in games if g.get("outcome") == "our_win")
    losses = sum(1 for g in games if g.get("outcome") == "opp_win")
    draws = sum(1 for g in games if g.get("outcome") == "draw")
    bad = sum(1 for g in games if g.get("outcome") in ("crash", "illegal", "exception", "aborted"))
    n_decisive = wins + losses
    wr = wins / n_decisive if n_decisive else float("nan")
    lo, hi = wilson_ci(wins, n_decisive) if n_decisive else (float("nan"), float("nan"))
    return dict(n=len(games), wins=wins, losses=losses, draws=draws, bad=bad,
                n_decisive=n_decisive, wr=wr, wilson_lo=lo, wilson_hi=hi)


def run_mode(mode: str, n_games: int, workers: int, decks_arg: str | None, max_minutes: float) -> None:
    state = load_state(mode)

    if mode == "field":
        decks = [d.strip() for d in (decks_arg or "").split(",") if d.strip()] or _list_meta_decks()
        cells = state.setdefault("cells", {})
        for d in decks:
            cells.setdefault(d, [])
        tasks = build_tasks_field(n_games, state, decks)
    else:
        have = len(state.get("games", []))
        builder = build_tasks_sanity if mode == "sanity" else build_tasks_h2h
        tasks = builder(n_games, have)

    print(f"[cardemb_gate:{mode}] tasks to run: {len(tasks)} (workers={workers})", flush=True)

    if tasks:
        ctx = __import__("multiprocessing").get_context("spawn")
        deadline = time.time() + max_minutes * 60.0
        t0 = time.time()
        n_done = 0
        total_illegal = total_exc = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            it = iter(tasks)
            futures = {}
            for _ in range(min(workers, len(tasks))):
                t = next(it)
                futures[ex.submit(play_one, t)] = t
            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    t = futures.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = dict(**t, outcome="crash", error=repr(e), n_illegal=0, n_exceptions=1,
                                   moves=0, ms_median=0.0, ms_max=0.0,
                                   n_model_decisions=None, n_fallback_decisions=None, n_agent_exceptions=None)
                    if mode == "field":
                        state["cells"].setdefault(res["meta_deck"], []).append(res)
                    else:
                        state.setdefault("games", []).append(res)
                    save_state(mode, state)
                    n_done += 1
                    total_illegal += res.get("n_illegal", 0) or 0
                    total_exc += res.get("n_exceptions", 0) or 0
                    tag = cell_key(res)
                    print(f"  [{n_done}/{len(tasks)}] {tag:20s} first={'Y' if t['our_first'] else 'N'} "
                          f"outcome={res.get('outcome'):10s} moves={res.get('moves', 0):4d} "
                          f"ms/dec={res.get('ms_median', 0):.0f} model={res.get('n_model_decisions')} "
                          f"fallback={res.get('n_fallback_decisions')}", flush=True)
                    if time.time() < deadline:
                        try:
                            nt = next(it)
                            futures[ex.submit(play_one, nt)] = nt
                        except StopIteration:
                            pass
        elapsed = time.time() - t0
        print(f"[cardemb_gate:{mode}] done: {n_done} games, illegal={total_illegal} exc={total_exc} "
              f"({elapsed:.0f}s / {elapsed/60:.1f}min)", flush=True)

    # --- Report ---
    if mode == "field":
        decks = list(state.get("cells", {}).keys())
        per_deck = {d: cell_stats(state["cells"][d]) for d in decks}
        all_games = [g for gs in state["cells"].values() for g in gs]
        overall = cell_stats(all_games)
        report = dict(mode=mode, overall=overall, per_deck=per_deck,
                      total_illegal=sum(g.get("n_illegal", 0) or 0 for g in all_games),
                      total_exceptions=sum(g.get("n_exceptions", 0) or 0 for g in all_games))
    else:
        games = state.get("games", [])
        overall = cell_stats(games)
        ms_meds = [g["ms_median"] for g in games if g.get("ms_median")]
        model_decs = [g["n_model_decisions"] for g in games if g.get("n_model_decisions") is not None]
        fallback_decs = [g["n_fallback_decisions"] for g in games if g.get("n_fallback_decisions") is not None]
        agent_excs = [g["n_agent_exceptions"] for g in games if g.get("n_agent_exceptions") is not None]
        report = dict(
            mode=mode, overall=overall,
            total_illegal=sum(g.get("n_illegal", 0) or 0 for g in games),
            total_exceptions=sum(g.get("n_exceptions", 0) or 0 for g in games),
            ms_per_decision_median=_median(ms_meds) if ms_meds else 0.0,
            ms_per_decision_max=(max(g["ms_max"] for g in games) if games else 0.0),
            model_decisions_total=sum(model_decs) if model_decs else 0,
            fallback_decisions_total=sum(fallback_decs) if fallback_decs else 0,
            model_share=(sum(model_decs) / (sum(model_decs) + sum(fallback_decs))
                        if (model_decs and (sum(model_decs) + sum(fallback_decs)) > 0) else None),
            agent_internal_exceptions_total=sum(agent_excs) if agent_excs else 0,
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path(mode), "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\n=== {mode.upper()} REPORT ===")
    print(json.dumps(report, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x))
    print(f"Report: {report_path(mode)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sanity", "h2h", "field"], required=True)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--decks", type=str, default="", help="field mode only: comma-separated meta deck names (default: all 19)")
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        run_mode(args.mode, 0, args.workers, args.decks, args.max_minutes)
        return 0

    run_mode(args.mode, args.games, args.workers, args.decks, args.max_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
