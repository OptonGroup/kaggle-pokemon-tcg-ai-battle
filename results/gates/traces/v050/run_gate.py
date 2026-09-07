"""v050 full local gate runner (cg_native, multiprocess).

Generalizes out/b5_length_analysis/h2h_v049_v048_steps.py's proven
ProcessPoolExecutor+spawn pattern to:
  (a) arbitrary agent-vs-opponent "pool" jobs (WR of ONE submission vs a
      fixed opponent, seat-balanced, N games), and
  (b) direct h2h jobs between two of our own candidates.

Jobs are described in a JSON job-list file (see build_jobs() below / --jobs).
Each job: {tag, kind: "pool"|"h2h", agent, opponent (pool) | a,b (h2h),
n_games, seed_base, debug (bool)}.

debug=true jobs run WITHOUT multiprocessing (sequential, single worker) so
that [dbg] stderr from our submission's _dbg() calls can be captured
per-game into its own log file (results/gates/traces/v050/dbg/<tag>/g<i>.log)
without interleaving across processes. Used only for small instrumentation
slices (P0/P1 signal-grepping), not for the bulk WR numbers.

debug=false jobs (the bulk of the gate) run with ProcessPoolExecutor,
_DEBUG_LOG monkey-patched off on any loaded v049/v050_* module (only those
three define it) for speed, and no per-game logs kept (win/loss/moves only).

Usage:
    .venv/bin/python out/v050_gate/run_gate.py --jobs out/v050_gate/jobs_wr.json \\
        --workers 8 --out out/v050_gate/results_wr.json
    .venv/bin/python out/v050_gate/run_gate.py --jobs out/v050_gate/jobs_dbg.json \\
        --out out/v050_gate/results_dbg.json   # debug jobs, sequential
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SUBMISSIONS = os.path.join(ROOT, "submissions")
MAX_MOVES = 2000

SUBMISSION_DIRS = {
    "v049": "v049_bench_sticky_guard",
    "v050_p0": "v050_p0",
    "v050_p0p1": "v050_p0p1",
    "v007": "v007_crustle",
    "v009a": "v009a_crustle",
    "v017": "v017_alakazam_heur",
    "v018": "v018_grimmsnarl",
    "v024": "v024_cardemb_rl",
    "v033": "v033_plan_d_ppo",
    "v034": "v034_romanrozen_clone",
    "v036": "v036_archaludon_patched",
    "v039": "v039_crustle_megakanga",
    "v044": "v044_archaludon_abomasnow",
    "v046": "v046_final",
    "v016": "v016_alakazam",
}

# submodules that define a module-level _DEBUG_LOG flag we can toggle off
# for speed on bulk (non-instrumented) runs.
_HAS_DEBUG_FLAG = {"v049", "v050_p0", "v050_p0p1"}


def wilson(wins, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / den
    return (max(0.0, c - m), min(1.0, c + m))


def _load_submission_module(short_name, tag):
    dir_name = SUBMISSION_DIRS[short_name]
    sub_dir = os.path.join(SUBMISSIONS, dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    sys_path_snapshot = list(sys.path)
    cwd = os.getcwd()
    try:
        sys.path.insert(0, sub_dir)
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_gate_{short_name}_{tag}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.path[:] = sys_path_snapshot
    return mod


def _load_deck(short_name):
    dir_name = SUBMISSION_DIRS[short_name]
    mod_path = os.path.join(SUBMISSIONS, dir_name, "deck.csv")
    with open(mod_path) as f:
        return [int(x.strip()) for x in f if x.strip()]


_MOD_CACHE = {}


def _get_module_cached(short_name, role_tag):
    key = (short_name, role_tag)
    if key not in _MOD_CACHE:
        _MOD_CACHE[key] = _load_submission_module(short_name, role_tag)
    return _MOD_CACHE[key]


def _ensure_cg_native():
    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)


@contextlib.contextmanager
def _redirect_stderr_to_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_stderr_fd = os.dup(2)
    with open(path, "w") as f:
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)


def _play_one_game(agent_mod, opp_mod, agent_deck, opp_deck, agent_is_left, track_opp_hand=False):
    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class

    if agent_is_left:
        deck0, deck1 = agent_deck, opp_deck
        agent0, agent1 = agent_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, agent_deck
        agent0, agent1 = opp_mod.agent, agent_mod.agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(outcome="aborted", moves=0)

    agent_idx = 0 if agent_is_left else 1
    opp_idx = 1 - agent_idx
    max_opp_hand_seen = 0
    end_our_bench_empty = None  # our bench count at game end (last observed on our turn)

    n_moves = 0
    exc_side = None
    winner = None
    try:
        while n_moves < MAX_MOVES:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your == 0 else agent1
            is_our_turn = (your == 0 and agent_is_left) or (your == 1 and not agent_is_left)
            if track_opp_hand and obs.current is not None:
                try:
                    players = obs.current.players
                    opp_hc = players[opp_idx].handCount
                    if opp_hc is not None and opp_hc > max_opp_hand_seen:
                        max_opp_hand_seen = opp_hc
                    my_bench = players[agent_idx].bench or []
                    end_our_bench_empty = (len([p for p in my_bench if p]) == 0)
                except Exception:
                    pass
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                exc_side = "agent" if is_our_turn else "opp"
                winner = 1 - your
                break
            obs_dict = battle_select(choice)
            n_moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    if winner is None or winner == 2:
        outcome = "draw" if winner == 2 else "aborted"
    else:
        agent_side = 0 if agent_is_left else 1
        outcome = "win" if winner == agent_side else "loss"
    res = dict(outcome=outcome, moves=n_moves, exc_side=exc_side)
    if track_opp_hand:
        res["max_opp_hand_seen"] = max_opp_hand_seen
        res["end_our_bench_empty"] = end_our_bench_empty
    return res


def _worker_task(task):
    """Runs in a ProcessPoolExecutor worker. task carries everything needed
    to load modules (cached at process level across tasks) and play one game."""
    _ensure_cg_native()
    kind = task["kind"]
    if kind == "pool":
        agent_short = task["agent"]
        opp_short = task["opponent"]
        agent_mod = _get_module_cached(agent_short, "agent")
        opp_mod = _get_module_cached(opp_short, "opp")
    else:  # h2h
        agent_short = task["a"]
        opp_short = task["b"]
        agent_mod = _get_module_cached(agent_short, "a")
        opp_mod = _get_module_cached(opp_short, "b")

    if not task.get("debug", False):
        for short, mod in ((agent_short, agent_mod), (opp_short, opp_mod)):
            if short in _HAS_DEBUG_FLAG:
                mod._DEBUG_LOG = False

    agent_deck = _load_deck(agent_short)
    opp_deck = _load_deck(opp_short)

    random.seed(task["seed"])

    track = task.get("track", False) or task.get("debug", False)
    if task.get("debug", False) and task.get("log_path"):
        with _redirect_stderr_to_file(task["log_path"]):
            print(f"=== GAME g={task['g']} agent={agent_short} opp={opp_short} "
                  f"agent_is_left={task['agent_is_left']} seed={task['seed']} ===", file=sys.stderr)
            res = _play_one_game(agent_mod, opp_mod, agent_deck, opp_deck, task["agent_is_left"],
                                  track_opp_hand=track)
            print(f"=== END outcome={res['outcome']} moves={res['moves']} "
                  f"max_opp_hand_seen={res.get('max_opp_hand_seen')} "
                  f"end_our_bench_empty={res.get('end_our_bench_empty')} ===", file=sys.stderr)
    else:
        res = _play_one_game(agent_mod, opp_mod, agent_deck, opp_deck, task["agent_is_left"],
                              track_opp_hand=track)

    res["tag"] = task["tag"]
    res["g"] = task["g"]
    res["agent_is_left"] = task["agent_is_left"]
    return res


def expand_job_tasks(job):
    n = job["n_games"]
    seed_base = job.get("seed_base", 20260721)
    tasks = []
    for g in range(n):
        task = {
            **{k: v for k, v in job.items() if k not in ("n_games", "seed_base")},
            "g": g,
            "seed": seed_base + g,
            "agent_is_left": (g % 2 == 0),
        }
        if job.get("debug", False):
            task["log_path"] = os.path.join(
                ROOT, "results", "gates", "traces", "v050", "dbg", job["tag"], f"g{g:04d}.log")
        tasks.append(task)
    return tasks


def run_jobs(jobs, workers, out_path):
    all_tasks = []
    for job in jobs:
        all_tasks.extend(expand_job_tasks(job))

    debug_tasks = [t for t in all_tasks if t.get("debug", False)]
    bulk_tasks = [t for t in all_tasks if not t.get("debug", False)]

    results_by_tag = {}

    def _record(res):
        results_by_tag.setdefault(res["tag"], []).append(res)

    t0 = time.time()
    if bulk_tasks:
        ctx = multiprocessing.get_context("spawn")
        print(f"[gate] {len(bulk_tasks)} bulk tasks, workers={workers}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_worker_task, t): t for t in bulk_tasks}
            n_done = 0
            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for f in done:
                    t = futures.pop(f)
                    try:
                        res = f.result()
                    except Exception as e:
                        res = dict(outcome="aborted", moves=0, tag=t["tag"], g=t["g"],
                                   agent_is_left=t["agent_is_left"], error=str(e))
                    _record(res)
                    n_done += 1
                    if n_done % 200 == 0:
                        print(f"  ... {n_done}/{len(bulk_tasks)} bulk done ({time.time()-t0:.0f}s)", flush=True)
    if debug_tasks:
        print(f"[gate] {len(debug_tasks)} debug tasks (sequential, single-process)", flush=True)
        for i, t in enumerate(debug_tasks):
            res = _worker_task(t)
            _record(res)
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(debug_tasks)} debug done ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"[gate] all done in {elapsed:.0f}s", flush=True)

    summary = {}
    for job in jobs:
        tag = job["tag"]
        rows = results_by_tag.get(tag, [])
        wins = sum(1 for r in rows if r["outcome"] == "win")
        losses = sum(1 for r in rows if r["outcome"] == "loss")
        draws = sum(1 for r in rows if r["outcome"] == "draw")
        aborted = sum(1 for r in rows if r["outcome"] == "aborted")
        n = wins + losses + draws
        we = wins + 0.5 * draws
        wr = we / n if n else float("nan")
        lo, hi = wilson(we, n)
        moves = [r["moves"] for r in rows if r["outcome"] in ("win", "loss")]
        summary[tag] = {
            "job": job, "n": n, "wins": wins, "losses": losses, "draws": draws,
            "aborted": aborted, "win_rate": wr, "wilson95_lo": lo, "wilson95_hi": hi,
            "avg_moves": (sum(moves) / len(moves)) if moves else 0.0,
        }
        loss_rows = [r for r in rows if r["outcome"] == "loss" and "end_our_bench_empty" in r]
        if loss_rows:
            n_bench_empty_losses = sum(1 for r in loss_rows if r.get("end_our_bench_empty"))
            summary[tag]["end_bench_empty_in_losses"] = n_bench_empty_losses
            summary[tag]["end_bench_empty_in_losses_n"] = len(loss_rows)
            summary[tag]["end_bench_empty_in_losses_rate"] = n_bench_empty_losses / len(loss_rows)
        hand_rows = [r for r in rows if "max_opp_hand_seen" in r]
        if hand_rows:
            ge8 = [r for r in hand_rows if (r.get("max_opp_hand_seen") or 0) >= 8]
            summary[tag]["games_opp_hand_ge8"] = len(ge8)
            summary[tag]["games_opp_hand_ge8_n"] = len(hand_rows)
        print(f"[result] {tag}: n={n} WR={wr:.3f} [{lo:.3f},{hi:.3f}] "
              f"({wins}W-{losses}L-{draws}D, aborted={aborted}) avg_moves={summary[tag]['avg_moves']:.1f}",
              flush=True)
        if loss_rows:
            print(f"          end_bench_empty_in_losses={summary[tag]['end_bench_empty_in_losses']}/"
                  f"{summary[tag]['end_bench_empty_in_losses_n']} "
                  f"({summary[tag]['end_bench_empty_in_losses_rate']:.2f})", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "elapsed_s": elapsed, "n_tasks": len(all_tasks)}, f, indent=2)
    print(f"[gate] wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="path to JSON job-list file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    run_jobs(jobs, args.workers, args.out)


if __name__ == "__main__":
    sys.exit(main())
