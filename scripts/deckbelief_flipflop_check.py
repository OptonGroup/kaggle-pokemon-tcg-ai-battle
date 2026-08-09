"""deckbelief_flipflop_check.py — counts detect_matchup() label transitions
per game for v047_deckbelief vs v017_alakazam_heur.

Baseline: live forensic on v046_judgefix (structurally identical board-only
detect_matchup logic to v044/v047) caught 39 alakazam<->generic transitions
across 20 games vs the same opponent (see
memory project_v046_judgefix_debug_logging_added.md) -- pure mercy/flicker
with no new information (the `seen` Pokemon list didn't change between
transitions), caused by detect_matchup() reverting to "generic" the instant
the last identifying Pokemon leaves the board.

v047's detect_matchup() persists via a module-level `_last_matchup` global
(see submissions/v047_deckbelief/main.py) -- this script measures that
global's value after each of OUR OWN decisions across N games and counts
how many times it changes, to confirm the fix actually holds on the real
engine (not just in isolated unit logic).

Usage:
    .venv/bin/python scripts/deckbelief_flipflop_check.py --games 20
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MAX_MOVES_PER_GAME = 2000
BASELINE_TRANSITIONS = 39
BASELINE_GAMES = 20


def _load_submission_module(sub_dir_name: str):
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_flipflop_{sub_dir_name}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def play_one_game(player_mod, opp_mod, player_deck, opp_deck, our_first: bool, seed: int) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class

    random.seed(seed)
    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = opp_mod.agent, player_mod.agent
    our_side = 0 if our_first else 1

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return {"outcome": "crash", "transitions": 0, "moves": 0, "matchup_sequence": []}

    move_count = 0
    winner = None
    n_exc = 0
    matchup_sequence = []
    prev_matchup = None
    transitions = 0
    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exc += 1
                break
            if your_index == our_side:
                cur = getattr(player_mod, "_last_matchup", None)
                if cur is not None:
                    matchup_sequence.append(cur)
                    if prev_matchup is not None and cur != prev_matchup:
                        transitions += 1
                    prev_matchup = cur
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    if n_exc > 0 or winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return {"outcome": outcome, "transitions": transitions, "moves": move_count,
            "matchup_sequence": matchup_sequence}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="v047_deckbelief")
    ap.add_argument("--opponent", default="v017_alakazam_heur")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260719)
    args = ap.parse_args()

    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)

    player_mod = _load_submission_module(args.candidate)
    opp_mod = _load_submission_module(args.opponent)
    player_deck = _load_deck_csv(os.path.join(ROOT, "submissions", args.candidate, "deck.csv"))
    opp_deck = _load_deck_csv(os.path.join(ROOT, "submissions", args.opponent, "deck.csv"))

    total_transitions = 0
    results = []
    for i in range(args.games):
        # Reset the candidate module's per-game globals the way agent() itself
        # does on a deck-select call (sel is None) -- battle_start triggers
        # that path via agent0/agent1's first call, so no extra reset needed
        # here; just play the game.
        res = play_one_game(player_mod, opp_mod, player_deck, opp_deck,
                             our_first=(i % 2 == 0), seed=args.seed + i)
        total_transitions += res["transitions"]
        results.append(res)
        print(f"  game {i+1:2d}/{args.games}: outcome={res['outcome']:10s} "
              f"transitions={res['transitions']:3d} moves={res['moves']}")

    print(f"\n=== {args.candidate} vs {args.opponent}, N={args.games} games ===")
    print(f"Total matchup-label transitions: {total_transitions}")
    print(f"Baseline (v046_judgefix, board-only detect_matchup): "
          f"{BASELINE_TRANSITIONS} transitions / {BASELINE_GAMES} games")
    print(f"Pass bar: strictly less than {BASELINE_TRANSITIONS}")
    passed = total_transitions < BASELINE_TRANSITIONS
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
