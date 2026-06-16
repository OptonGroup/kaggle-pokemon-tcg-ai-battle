"""Self-play simulation harness.

NOTE: libcg.so is a Linux x86_64 shared library — won't load on macOS ARM.
Run simulations on Kaggle notebooks or inside a Linux Docker container.
"""
import sys
import os

# Add src to path so cg module resolves
sys.path.insert(0, os.path.dirname(__file__))

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class


def run_game(agent0, agent1, deck0: list[int], deck1: list[int]) -> int:
    """Run a single game between two agents. Returns winner index (0/1) or -1 for draw."""
    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"Battle failed to start: errorType={start_data.errorType}")

    try:
        while True:
            obs = to_observation_class(obs_dict)

            if obs.current and obs.current.result != -1:
                return obs.current.result

            your_index = obs.current.yourIndex if obs.current else 0
            agent = agent0 if your_index == 0 else agent1

            if obs.select is None:
                select = agent(obs_dict)  # initial deck selection
            else:
                select = agent(obs_dict)

            obs_dict = battle_select(select)
    finally:
        battle_finish()


def run_tournament(agent0, agent1, deck0: list[int], deck1: list[int], n_games: int = 100) -> dict:
    wins = [0, 0]
    draws = 0
    for i in range(n_games):
        # alternate who goes first
        if i % 2 == 0:
            result = run_game(agent0, agent1, deck0, deck1)
            if result == 0:
                wins[0] += 1
            elif result == 1:
                wins[1] += 1
            else:
                draws += 1
        else:
            result = run_game(agent1, agent0, deck1, deck0)
            if result == 0:
                wins[1] += 1
            elif result == 1:
                wins[0] += 1
            else:
                draws += 1
    total = n_games - draws
    return {
        "wins_0": wins[0],
        "wins_1": wins[1],
        "draws": draws,
        "winrate_0": wins[0] / total if total else 0.5,
    }
