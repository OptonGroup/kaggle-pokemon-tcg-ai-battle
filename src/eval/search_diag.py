"""Diagnostic script: confirm whether search_begin actually runs during MAIN decisions.

Plays several games (search_agent vs heuristic), instruments:
  - How many MAIN decisions reached _search_decision
  - How often search_begin SUCCEEDED vs raised (and the exception message)
  - Mean options evaluated per MAIN decision
  - Actual wall-time per MAIN decision
  - Fraction using search vs heuristic fallback

Run inside Docker:
    python src/eval/search_diag.py [n_games] [seed]
"""
import os
import sys
import time
import random

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, SelectContext, search_begin, search_end
import agents.heuristic_agent as H

DECK = list(H.my_deck)

# Counters
_main_decisions = 0
_search_begin_ok = 0
_search_begin_fail = 0
_search_begin_errors = {}  # error message -> count
_options_evaluated = []    # list of int (per decision, how many options got a value)
_main_times = []           # wall times for MAIN decisions only
_search_used_count = 0     # MAIN decisions where search returned a result (not None)
_heuristic_fallback_count = 0  # MAIN decisions that fell back to heuristic


def instrumented_search_agent(obs_dict: dict) -> list[int]:
    """Instrumented version of the search agent's decide() for one decision."""
    global _main_decisions, _search_begin_ok, _search_begin_fail
    global _search_used_count, _heuristic_fallback_count

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK

    sel = obs.select
    context = sel.context

    t0 = time.perf_counter()
    is_main = (
        context == SelectContext.MAIN
        and len(sel.option) >= 2
        and sel.minCount == 1 and sel.maxCount == 1
        and obs_dict.get("search_begin_input") is not None
    )

    if is_main:
        _main_decisions += 1
        # Try up to 4 determinizations, count successes/failures
        import agents.search_agent as SA
        rng = random.Random(_main_decisions * 1000003)
        n_opts = len(sel.option)
        scores = [0.0] * n_opts
        counts = [0] * n_opts
        K = 4
        deadline = t0 + 2.75  # ~budget

        for k in range(K):
            if time.perf_counter() >= deadline:
                break
            det = SA.build_determinization(obs, rng)
            yd, yp, od, op, oh, oa = det
            try:
                root = search_begin(obs, yd, yp, od, op, oh, oa)
                _search_begin_ok += 1
                # try first option only (enough to confirm it works)
                from cg.api import search_step
                try:
                    child = search_step(root.searchId, [0])
                    scores[0] += 0.0
                    counts[0] += 1
                except Exception as e:
                    pass
                search_end()
            except Exception as e:
                _search_begin_fail += 1
                msg = f"{type(e).__name__}: {str(e)[:80]}"
                _search_begin_errors[msg] = _search_begin_errors.get(msg, 0) + 1
                try:
                    search_end()
                except Exception:
                    pass

        opts_eval = sum(1 for c in counts if c > 0)
        _options_evaluated.append(opts_eval)
        dt = time.perf_counter() - t0
        _main_times.append(dt)

        if _search_begin_ok > 0 and opts_eval > 0:
            _search_used_count += 1
        else:
            _heuristic_fallback_count += 1

    # Always use heuristic for actual move (this is diagnostics only)
    return H.agent(obs_dict)


def run_game():
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
                choice = instrumented_search_agent(obs_dict)
            else:
                choice = H.agent(obs_dict)
            obs_dict = battle_select(choice)
    finally:
        battle_finish()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    random.seed(seed)

    print(f"=== search_diag: {n} games, seed={seed} ===")
    for i in range(n):
        try:
            result = run_game()
            print(f"  game {i}: result={result}")
        except Exception as e:
            print(f"  game {i}: ERROR {e}")

    print()
    print("=== DIAGNOSIS ===")
    print(f"  MAIN decisions reached       : {_main_decisions}")
    print(f"  search_begin OK              : {_search_begin_ok}")
    print(f"  search_begin FAILED          : {_search_begin_fail}")
    print(f"  (note: 4 attempts per MAIN decision, so OK+FAIL = _main_decisions*4)")
    if _search_begin_errors:
        print("  Error messages:")
        for msg, cnt in sorted(_search_begin_errors.items(), key=lambda x: -x[1]):
            print(f"    [{cnt}x] {msg}")
    print(f"  Mean opts evaluated/decision : "
          f"{sum(_options_evaluated)/max(1,len(_options_evaluated)):.2f}")
    print(f"  MAIN decisions w/ search     : {_search_used_count}")
    print(f"  MAIN decisions w/ fallback   : {_heuristic_fallback_count}")
    if _main_times:
        print(f"  Mean MAIN decision time      : {1000*sum(_main_times)/len(_main_times):.2f} ms")
        print(f"  Max  MAIN decision time      : {1000*max(_main_times):.1f} ms")
    print()
    if _search_begin_fail > 0 and _search_begin_ok == 0:
        print("CONCLUSION: search_begin is ALWAYS failing => search never ran!")
    elif _search_begin_ok > 0:
        print(f"CONCLUSION: search_begin succeeded {_search_begin_ok} times "
              f"(failed {_search_begin_fail} times).")
    else:
        print("CONCLUSION: No MAIN decisions reached => game ended too quickly?")


if __name__ == "__main__":
    main()
