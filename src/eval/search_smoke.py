"""Quick smoke test for the dev search_agent: run a few games vs the heuristic,
report wins, exceptions, and per-move timing. Runs in Docker.

Usage: python src/eval/search_smoke.py [n_games] [seed]
"""
import os
import sys
import time
import random

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class

import agents.search_agent as SA
import agents.heuristic_agent as H

DECK = list(SA.my_deck)


def run_game(a0, a1):
    obs_dict, start = battle_start(DECK, DECK)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed errorType={start.errorType}")
    move_times = []
    exceptions = []
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                return obs.current.result, move_times, exceptions, obs.current.turn
            yi = obs.current.yourIndex if obs.current else 0
            agent = a0 if yi == 0 else a1
            t0 = time.perf_counter()
            try:
                choice = agent(obs_dict)
            except Exception as e:
                exceptions.append(f"{type(e).__name__}: {e}")
                # never-throw wrappers should prevent this; record and pick safe
                sel = obs.select
                choice = list(range(max(1, sel.minCount))) if sel else DECK
            move_times.append(time.perf_counter() - t0)
            obs_dict = battle_select(choice)
    finally:
        battle_finish()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    random.seed(seed)

    sa = SA.agent
    he = H.agent

    sa_wins = 0
    he_wins = 0
    draws = 0
    all_times = []
    max_time = 0.0
    total_exc = 0
    per_game_search_time = []

    for i in range(n):
        # reset search agent game state cleanly each game
        SA._reset_game_state()
        if i % 2 == 0:
            res, times, exc, turns = run_game(sa, he)
            sa_side = 0
        else:
            res, times, exc, turns = run_game(he, sa)
            sa_side = 1
        all_times.extend(times)
        max_time = max(max_time, max(times) if times else 0)
        total_exc += len(exc)
        if res == 2 or res == -1:
            draws += 1
        elif res == sa_side:
            sa_wins += 1
        else:
            he_wins += 1
        print(f"  game {i}: winner={res} sa_side={sa_side} turns={turns} "
              f"moves={len(times)} max_move={max(times) if times else 0:.3f}s "
              f"game_time_used={SA._time_spent:.2f}s exc={len(exc)}")
        if exc:
            for e in exc[:3]:
                print(f"    EXC: {e}")

    print()
    print(f"=== SMOKE ({n} games) ===")
    print(f"  search_agent wins : {sa_wins}")
    print(f"  heuristic wins    : {he_wins}")
    print(f"  draws             : {draws}")
    print(f"  total exceptions  : {total_exc}")
    print(f"  mean move time    : {1000*sum(all_times)/len(all_times):.2f} ms")
    print(f"  max  move time    : {1000*max_time:.1f} ms")


if __name__ == "__main__":
    main()
