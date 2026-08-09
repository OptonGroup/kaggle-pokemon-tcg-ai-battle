#!/usr/bin/env python3
"""cp_selfplay_smoke.py — 50-game-per-deck self-play smoke test for the committed-plan
Phase 2 submissions (submissions/cp_lopunny, submissions/cp_ogerpon_teal).

Adapted from the scratchpad prototype cp_wrapper_smoke.py (already validated at 4 games/
deck). Extends to 50 games/deck (100 total), adds a progress log every 10 games, and wraps
every single game in a multiprocessing timeout so a hang inside cg_native cannot stall the
whole run (per project note: threading previously SIGSEGV'd against cg_native, so
multiprocessing is used instead of a thread-based timeout).

Reuses the exact submission-loading pattern from scripts/gate_scout_guard.py's
_load_submission_module/_play_one_game (chdir into the submission dir, spec_from_file_location
with __file__ set), and cg_native.game for the engine.

Usage:
  .venv/bin/python scripts/cp_selfplay_smoke.py                 # 50 games/deck, CP_DEBUG=0
  .venv/bin/python scripts/cp_selfplay_smoke.py --audit N        # extra N debug-logged games
                                                                  # per deck, log saved to file
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
import time

REPO = "/Users/max/kaggle-pokemon-tcg-ai-battle"
SRC = os.path.join(REPO, "src")
SCRIPTS = os.path.join(REPO, "scripts")
for _p in (SRC, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_MOVES = 400
N_GAMES = 50
GAME_TIMEOUT_S = 60

SUBMISSIONS = ["cp_lopunny", "cp_ogerpon_teal"]


def _load_submission_module(sub_dir: str, tag: str):
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"missing {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_cpsmoke_{tag}"
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _is_legal(choice, sel):
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = sel.get("minCount")
    max_c = sel.get("maxCount")
    min_c = 1 if min_c is None else int(min_c)
    max_c = 1 if max_c is None else int(max_c)
    if not isinstance(choice, list):
        return False
    if not (min_c <= len(choice) <= max_c):
        return False
    if len(set(choice)) != len(choice):
        return False
    return all(isinstance(i, int) and 0 <= i < n_opts for i in choice)


def _play_one(cand_agent, cand_deck, opp_deck, cand_first, greedy_meta_agent, battle_start, battle_finish, battle_select, to_observation_class):
    # New-episode boundary: reset the module-level `_plan` cache (see committed_plan's own
    # docstring point 1) before each fresh game, matching the real Kaggle new-episode
    # contract (agent() called with obs_dict["select"]=None at the start of each episode).
    cand_agent({"select": None})
    if cand_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, greedy_meta_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = greedy_meta_agent, cand_agent

    obs_dict, _start = battle_start(deck0, deck1)
    if obs_dict is None:
        return {"outcome": "crash_start"}

    moves = 0
    n_exceptions = 0
    exc_side = None
    winner = None
    try:
        while moves < MAX_MOVES:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            sel = obs_dict.get("select")
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                n_exceptions += 1
                exc_side = "cand" if (your_index == 0) == cand_first else "opp"
                print(f"    [smoke] agent EXCEPTION side={exc_side}: {e!r}", flush=True)
                break
            if not _is_legal(choice, sel):
                print(f"    [smoke] ILLEGAL choice={choice} sel={sel}", flush=True)
                break
            obs_dict = battle_select(choice)
            moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    cand_side = 0 if cand_first else 1
    return {
        "outcome": "ok" if (winner is not None or n_exceptions == 0) else "exception",
        "winner": winner,
        "cand_won": (winner == cand_side) if winner is not None else None,
        "moves": moves,
        "n_exceptions": n_exceptions,
        "exc_side": exc_side,
    }


# ── multiprocessing worker globals (populated once per worker process by _worker_init) ────
_W = {}


def _worker_init(sub_dir: str, tag: str, debug: bool, log_path: str | None):
    if debug:
        os.environ["CP_DEBUG"] = "1"
        if log_path:
            f = open(log_path, "a", buffering=1)
            sys.stderr = f
            sys.stdout = f
    else:
        os.environ["CP_DEBUG"] = "0"

    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class
    from gate_scout_guard import greedy_meta_agent

    mod = _load_submission_module(sub_dir, tag)
    _W["agent"] = mod.agent
    _W["cp"] = mod.committed_plan
    _W["deck"] = mod.agent({"select": None})
    _W["greedy_meta_agent"] = greedy_meta_agent
    _W["battle_start"] = battle_start
    _W["battle_finish"] = battle_finish
    _W["battle_select"] = battle_select
    _W["to_observation_class"] = to_observation_class
    assert len(_W["deck"]) == 60, f"deck len {len(_W['deck'])}"


def _worker_play(cand_first: bool):
    cp = _W["cp"]
    cp.STATS["plan_built"] = 0
    cp.STATS["plan_missing"] = 0
    cp.STATS["generic_fallback"] = 0
    cp.STATS["exceptions"] = 0
    deck = _W["deck"]
    r = _play_one(
        _W["agent"], deck, deck, cand_first,
        _W["greedy_meta_agent"], _W["battle_start"], _W["battle_finish"],
        _W["battle_select"], _W["to_observation_class"],
    )
    r["plan_built"] = cp.STATS["plan_built"]
    r["plan_missing"] = cp.STATS["plan_missing"]
    r["generic_fallback"] = cp.STATS["generic_fallback"]
    r["cp_exceptions"] = cp.STATS["exceptions"]
    return r


def _new_pool(sub_dir: str, tag: str, debug: bool, log_path: str | None):
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=1, initializer=_worker_init, initargs=(sub_dir, tag, debug, log_path))


def run_deck(tag: str, n_games: int, debug: bool = False, log_path: str | None = None):
    sub_dir = os.path.join(REPO, "submissions", tag)
    print(f"\n=== {tag} ({sub_dir}) n_games={n_games} debug={debug} ===", flush=True)

    pool = _new_pool(sub_dir, tag, debug, log_path)
    results = []
    n_timeouts = 0
    n_worker_errors = 0
    t0 = time.time()

    i = 0
    while i < n_games:
        cand_first = (i % 2 == 0)
        try:
            async_res = pool.apply_async(_worker_play, (cand_first,))
            r = async_res.get(timeout=GAME_TIMEOUT_S)
            results.append(r)
        except mp.TimeoutError:
            n_timeouts += 1
            print(f"[smoke] {tag} game {i} TIMEOUT after {GAME_TIMEOUT_S}s -- killing pool, recreating", flush=True)
            pool.terminate()
            pool.join()
            results.append({"outcome": "timeout", "plan_built": 0, "plan_missing": 0,
                             "generic_fallback": 0, "cp_exceptions": 0, "n_exceptions": 1})
            pool = _new_pool(sub_dir, tag, debug, log_path)
        except Exception as e:
            n_worker_errors += 1
            print(f"[smoke] {tag} game {i} WORKER ERROR: {e!r} -- killing pool, recreating", flush=True)
            pool.terminate()
            pool.join()
            results.append({"outcome": "worker_error", "plan_built": 0, "plan_missing": 0,
                             "generic_fallback": 0, "cp_exceptions": 0, "n_exceptions": 1})
            pool = _new_pool(sub_dir, tag, debug, log_path)

        if (i + 1) % 10 == 0 or i == n_games - 1:
            elapsed = time.time() - t0
            print(f"[smoke] {tag} game {i + 1}/{n_games} done ({elapsed:.1f}s elapsed)", flush=True)
        i += 1

    pool.terminate()
    pool.join()

    total_plan_built = sum(r.get("plan_built", 0) for r in results)
    total_plan_missing = sum(r.get("plan_missing", 0) for r in results)
    total_generic_fallback = sum(r.get("generic_fallback", 0) for r in results)
    total_cp_exc = sum(r.get("cp_exceptions", 0) for r in results)
    total_agent_exc = sum(r.get("n_exceptions", 0) for r in results)
    n_ok = sum(1 for r in results if r.get("outcome") == "ok")

    summary = {
        "tag": tag,
        "n_games": n_games,
        "n_ok": n_ok,
        "n_timeouts": n_timeouts,
        "n_worker_errors": n_worker_errors,
        "plan_built": total_plan_built,
        "plan_missing": total_plan_missing,
        "generic_fallback": total_generic_fallback,
        "cp_exceptions": total_cp_exc,
        "agent_exceptions": total_agent_exc,
    }
    verdict = "PASS" if (
        n_timeouts == 0 and n_worker_errors == 0
        and total_cp_exc == 0 and total_agent_exc == 0
        and total_plan_built > 0 and n_ok == n_games
    ) else "FAIL"
    summary["verdict"] = verdict
    print(f"[smoke] {tag} SUMMARY: {summary}", flush=True)
    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=N_GAMES)
    parser.add_argument("--audit", type=int, default=0, help="extra CP_DEBUG=1 games per deck, logged to file")
    parser.add_argument("--audit-log-dir", type=str, default="/tmp")
    args = parser.parse_args()

    all_summaries = []
    for tag in SUBMISSIONS:
        summary, _ = run_deck(tag, args.n_games, debug=False, log_path=None)
        all_summaries.append(summary)

    print("\n=== MAIN RUN FINAL ===", flush=True)
    for s in all_summaries:
        print(s, flush=True)

    if args.audit > 0:
        print(f"\n=== AUDIT RUNS (CP_DEBUG=1, {args.audit} games/deck) ===", flush=True)
        for tag in SUBMISSIONS:
            log_path = os.path.join(args.audit_log_dir, f"cp_audit_{tag}.log")
            open(log_path, "w").close()  # truncate
            run_deck(tag, args.audit, debug=True, log_path=log_path)
            print(f"[smoke] audit log for {tag} -> {log_path}", flush=True)
