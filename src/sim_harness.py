"""Simulation harness for self-play tournaments.

Extends src/simulate.py with:
- Wilson 95% confidence interval on winrate
- Mean per-move wall-time measurement
- Q-LEGAL check: verifies the engine rejects illegal decks

NOTE: libcg.so is a Linux x86_64 shared library — won't load on macOS ARM.
Run inside Docker (./scripts/sim.sh) or a Kaggle notebook.
"""
import sys
import os
import math
import time
import random

# Ensure src/ is on the path so `import cg` and `import agents.*` resolve
sys.path.insert(0, os.path.dirname(__file__))

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class

# Re-use existing game loop from simulate.py
from simulate import run_game


# ---------------------------------------------------------------------------
# Wilson score interval (two-sided, 95%)
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) Wilson 95% CI for a proportion.

    Args:
        wins: number of successes.
        n: total trials (wins + losses, NOT draws).
        z: z-score for desired confidence (1.96 → 95%).

    Returns:
        (lower_bound, upper_bound) as floats in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Timed game loop
# ---------------------------------------------------------------------------

def run_game_timed(agent0, agent1, deck0: list[int], deck1: list[int],
                   overage_bank: float = 600.0) -> tuple[int, int, float]:
    """Run a single game and return (winner, move_count, mean_move_time_seconds).

    winner: 0, 1, or -1 (draw).

    overage_bank mirrors the live per-episode time bank each agent may spend
    (600s on Kaggle). Exhausting it loses the game, as DeadlineExceeded does live.
    """
    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(
            f"Battle failed to start: errorPlayer={start_data.errorPlayer} "
            f"errorType={start_data.errorType}"
        )

    move_count = 0
    total_move_time = 0.0
    # The live engine bills each move against a per-episode bank and exposes what
    # is left as observation.remainingOverageTime (actTimeout=0, 600s bank). Time-
    # budgeted agents read it to pace themselves, so a harness that omits it makes
    # them behave as if the bank were empty and measures the wrong pilot.
    bank = {0: overage_bank, 1: overage_bank}

    try:
        while True:
            obs = to_observation_class(obs_dict)

            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                mean_time = total_move_time / move_count if move_count else 0.0
                return winner, move_count, mean_time

            your_index = obs.current.yourIndex if obs.current else 0
            agent = agent0 if your_index == 0 else agent1

            obs_dict["remainingOverageTime"] = bank[your_index]
            t0 = time.perf_counter()
            select = agent(obs_dict)
            dt = time.perf_counter() - t0
            total_move_time += dt
            bank[your_index] -= dt
            if bank[your_index] <= 0:
                # Live, this is DeadlineExceeded: the offender's reward is nulled.
                return (1 - your_index), move_count, (
                    total_move_time / move_count if move_count else 0.0
                )

            obs_dict = battle_select(select)
            move_count += 1
    finally:
        battle_finish()


# ---------------------------------------------------------------------------
# Public API: tournament()
# ---------------------------------------------------------------------------

def tournament(
    agentA,
    agentB,
    deckA: list[int],
    deckB: list[int],
    n: int = 100,
    seed: int | None = None,
) -> dict:
    """Run n games between agentA and agentB, alternating who goes first.

    Args:
        agentA: callable matching the competition agent signature.
        agentB: callable matching the competition agent signature.
        deckA: 60-card deck for agentA (list of card IDs).
        deckB: 60-card deck for agentB (list of card IDs).
        n: number of games to play.
        seed: optional random seed for reproducibility.

    Returns:
        dict with keys:
            wins_A, wins_B, draws, n_games,
            winrate_A (excl. draws),
            wilson_lo, wilson_hi  (95% CI on winrate_A, excl. draws),
            mean_move_time_s      (mean wall-time per move across all games)
    """
    if seed is not None:
        random.seed(seed)

    wins_A = 0
    wins_B = 0
    draws = 0
    total_moves = 0
    total_move_time = 0.0

    for i in range(n):
        if i % 2 == 0:
            # A goes first
            result, moves, mt = run_game_timed(agentA, agentB, deckA, deckB)
            total_moves += moves
            total_move_time += mt * moves
            if result == 0:
                wins_A += 1
            elif result == 1:
                wins_B += 1
            else:
                draws += 1
        else:
            # B goes first; flip result
            result, moves, mt = run_game_timed(agentB, agentA, deckB, deckA)
            total_moves += moves
            total_move_time += mt * moves
            if result == 0:
                wins_B += 1
            elif result == 1:
                wins_A += 1
            else:
                draws += 1

    decisive = wins_A + wins_B  # excluding draws
    winrate_A = wins_A / decisive if decisive else 0.5
    lo, hi = wilson_ci(wins_A, decisive)
    mean_move_time = total_move_time / total_moves if total_moves else 0.0

    return {
        "wins_A": wins_A,
        "wins_B": wins_B,
        "draws": draws,
        "n_games": n,
        "winrate_A": winrate_A,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_move_time_s": mean_move_time,
    }


# ---------------------------------------------------------------------------
# Q-LEGAL check: verify the engine rejects an illegal deck
# ---------------------------------------------------------------------------

def check_illegal_deck() -> None:
    """Verify battle_start rejects an obviously illegal deck.

    Test 1: wrong length (not 60) — should raise ValueError before even
            reaching the engine.
    Test 2: 60 copies of card ID 1 (Basic Water Energy) — valid length but
            no Basic Pokémon, which is illegal per TCG rules. The engine
            should return errorPlayer >= 0 (not -1) and a null battlePtr.
    """
    print("\n--- Q-LEGAL check ---")

    # Test 1: wrong deck length
    try:
        battle_start([1] * 59, [1] * 60)
        print("FAIL: expected ValueError for wrong-length deck")
    except ValueError as e:
        print(f"PASS (wrong length): engine raised ValueError: {e}")

    # Test 2: 60 copies of one non-Pokemon card (no Basic Pokemon in deck)
    illegal_deck = [1] * 60  # card ID 1 = Basic {G} Energy (no Pokémon)
    legal_deck_path = os.path.join(
        os.path.dirname(__file__), "..", "submissions", "v001_random", "deck.csv"
    )
    with open(legal_deck_path) as f:
        legal_deck = [int(line.strip()) for line in f if line.strip()]

    obs_dict, start_data = battle_start(illegal_deck, legal_deck)
    if obs_dict is None:
        print(
            f"PASS (no Basic Pokemon): engine rejected illegal deck — "
            f"errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}"
        )
        # Must still finish the failed battle to free resources
        try:
            battle_finish()
        except Exception:
            pass
    else:
        print(
            f"NOTE: engine did not reject 60×energy deck "
            f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType}); "
            f"game may proceed then error mid-setup"
        )
        try:
            battle_finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load agents from within src/agents/ (PYTHONPATH already includes src/)
    from agents.random_agent import agent as random_agent

    deck_path = os.path.join(
        os.path.dirname(__file__), "..", "submissions", "v001_random", "deck.csv"
    )
    with open(deck_path) as f:
        deck = [int(line.strip()) for line in f if line.strip()]

    # Accept optional CLI args: n_games seed
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"Running random-vs-random tournament: {n_games} games (seed={seed})")
    print(f"Deck: {deck_path} ({len(deck)} cards)")

    results = tournament(random_agent, random_agent, deck, deck, n=n_games, seed=seed)

    print()
    print(f"Results ({n_games} games):")
    print(f"  Agent A wins : {results['wins_A']}")
    print(f"  Agent B wins : {results['wins_B']}")
    print(f"  Draws        : {results['draws']}")
    print(f"  Win-rate A   : {results['winrate_A']:.3f}  "
          f"(Wilson 95% CI: [{results['wilson_lo']:.3f}, {results['wilson_hi']:.3f}])")
    print(f"  Mean move time: {results['mean_move_time_s']*1000:.3f} ms")

    # Q-LEGAL verification
    check_illegal_deck()


if __name__ == "__main__":
    main()
