"""Deeper diagnostic: measure actual rollout depth + time in the REAL search agent loop.

Run inside Docker:
    python src/eval/search_diag2.py [n_games] [seed]
"""
import os
import sys
import time
import random

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from cg.game import battle_start, battle_finish, battle_select
from cg.api import (
    to_observation_class, SelectContext, search_begin, search_end,
    search_step as engine_search_step,
)
import agents.heuristic_agent as H
import agents.search_agent as SA

DECK = list(H.my_deck)

# Deep rollout stats
_rollout_depths = []
_rollout_times = []
_full_decision_times = []  # time for entire MAIN decision (all K * n_opts rollouts)
_opts_per_decision = []
_k_completed = []  # how many determinizations completed per decision
_search_begin_ok = 0
_search_begin_fail = 0


def run_one_rollout(root, our_index, rng, deadline):
    """Run actual rollout, return (value, steps, time)."""
    t0 = time.perf_counter()
    steps = 0
    state = root
    while True:
        obs = state.observation
        if obs.current is not None and obs.current.result != -1:
            return SA._result_value(obs.current.result, our_index), steps, time.perf_counter()-t0
        sel = obs.select
        if sel is None:
            return SA._leaf_eval(obs, our_index), steps, time.perf_counter()-t0
        if steps >= SA.ROLLOUT_DEPTH_CAP or time.perf_counter() >= deadline:
            return SA._leaf_eval(obs, our_index), steps, time.perf_counter()-t0
        choice = SA._rollout_choice(obs, rng)
        state = engine_search_step(state.searchId, choice)
        steps += 1


def instrumented_decision(obs, rng, budget):
    """Run K determinizations + rollout per option, track stats."""
    global _search_begin_ok, _search_begin_fail
    sel = obs.select
    n_opts = len(sel.option)
    our_index = obs.current.yourIndex
    deadline = time.perf_counter() + budget

    t_dec_start = time.perf_counter()
    ks_completed = 0

    saved = SA._save_heuristic_globals()
    try:
        for k in range(SA.K_DETERMINIZATIONS):
            if time.perf_counter() >= deadline:
                break
            det = SA.build_determinization(obs, rng)
            yd, yp, od, op, oh, oa = det
            try:
                root = search_begin(obs, yd, yp, od, op, oh, oa)
                _search_begin_ok += 1
            except Exception as e:
                _search_begin_fail += 1
                try: search_end()
                except: pass
                continue

            try:
                for opt_idx in range(n_opts):
                    if time.perf_counter() >= deadline:
                        break
                    try:
                        child = engine_search_step(root.searchId, [opt_idx])
                    except Exception:
                        continue
                    obs_c = child.observation
                    if obs_c.current is not None and obs_c.current.result != -1:
                        val = SA._result_value(obs_c.current.result, our_index)
                        _rollout_depths.append(0)
                        _rollout_times.append(0.0)
                    else:
                        val, steps, rt = run_one_rollout(child, our_index, rng, deadline)
                        _rollout_depths.append(steps)
                        _rollout_times.append(rt)
                ks_completed += 1
            finally:
                try: search_end()
                except: pass
    finally:
        SA._restore_heuristic_globals(saved)

    _full_decision_times.append(time.perf_counter() - t_dec_start)
    _opts_per_decision.append(n_opts)
    _k_completed.append(ks_completed)


def instrumented_agent(obs_dict: dict) -> list[int]:
    """Run diagnostic but use heuristic for actual move."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        SA._reset_game_state()
        return DECK

    sel = obs.select
    context = sel.context

    is_main = (
        context == SelectContext.MAIN
        and len(sel.option) >= 2
        and sel.minCount == 1 and sel.maxCount == 1
        and obs_dict.get("search_begin_input") is not None
    )

    if is_main:
        budget = SA._per_decision_budget()
        if budget >= SA.MIN_BUDGET_S:
            rng = random.Random(SA._decisions_made * 1000003)
            instrumented_decision(obs, rng, budget)

    # Always use heuristic for the actual move
    return H.agent(obs_dict)


def run_game(game_idx):
    SA._reset_game_state()
    obs_dict, start = battle_start(DECK, DECK)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed errorType={start.errorType}")
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                return obs.current.result
            yi = obs.current.yourIndex if obs.current else 0
            if yi == 0:
                choice = instrumented_agent(obs_dict)
            else:
                choice = H.agent(obs_dict)
            obs_dict = battle_select(choice)
    finally:
        battle_finish()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    random.seed(seed)

    print(f"=== search_diag2: {n} games, seed={seed} ===")
    for i in range(n):
        try:
            result = run_game(i)
            print(f"  game {i}: result={result}, search_time_used={SA._time_spent:.2f}s")
        except Exception as e:
            import traceback
            print(f"  game {i}: ERROR {e}")
            traceback.print_exc()

    print()
    print("=== DEEP DIAGNOSIS ===")
    print(f"  search_begin OK              : {_search_begin_ok}")
    print(f"  search_begin FAILED          : {_search_begin_fail}")
    print(f"  MAIN decisions instrumented  : {len(_full_decision_times)}")

    if _rollout_depths:
        print(f"  Rollouts total               : {len(_rollout_depths)}")
        print(f"  Mean rollout depth (steps)   : {sum(_rollout_depths)/len(_rollout_depths):.1f}")
        print(f"  Max rollout depth (steps)    : {max(_rollout_depths)}")
        print(f"  Min rollout depth            : {min(_rollout_depths)}")
        print(f"  % that hit depth cap ({SA.ROLLOUT_DEPTH_CAP}) : "
              f"{100*sum(1 for d in _rollout_depths if d >= SA.ROLLOUT_DEPTH_CAP)/len(_rollout_depths):.1f}%")
        print(f"  Mean rollout wall time       : {1000*sum(_rollout_times)/len(_rollout_times):.1f} ms")
        print(f"  Max rollout wall time        : {1000*max(_rollout_times):.1f} ms")
    else:
        print("  No rollouts completed!")

    if _full_decision_times:
        print(f"  Mean MAIN decision time      : {1000*sum(_full_decision_times)/len(_full_decision_times):.1f} ms")
        print(f"  Max  MAIN decision time      : {1000*max(_full_decision_times):.1f} ms")
        print(f"  Mean K completed/decision    : {sum(_k_completed)/len(_k_completed):.1f}")
        print(f"  Mean opts/decision           : {sum(_opts_per_decision)/len(_opts_per_decision):.1f}")

    if _rollout_depths and _full_decision_times:
        # How many rollouts per decision on average?
        rp = len(_rollout_depths) / len(_full_decision_times)
        print(f"  Mean rollouts/MAIN decision  : {rp:.1f}")
        # Time budget usage
        if _full_decision_times:
            mean_dt = sum(_full_decision_times)/len(_full_decision_times)
            print(f"  Budget usage vs ~2.75s/dec   : {100*mean_dt/2.75:.1f}%")


if __name__ == "__main__":
    main()
