"""P1-2 validation script for v002_lucario.

Step 0: deck legality check — call battle_start(v002_deck, v001_deck) and assert non-None obs_dict.
Step 1: 20-game smoke test of v002 vs random; confirm ZERO exceptions.

Run from /workspace (PYTHONPATH includes /workspace/src):
    python src/eval/validate_v002.py
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class
from sim_harness import tournament

# ---------------------------------------------------------------------------
# Load decks
# ---------------------------------------------------------------------------

def load_deck(path: str) -> list[int]:
    with open(path) as f:
        cards = [int(line.strip()) for line in f if line.strip()]
    assert len(cards) == 60, f"Expected 60 cards, got {len(cards)} in {path}"
    return cards

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
v002_deck_path = os.path.join(REPO_ROOT, "submissions", "v002_lucario", "deck.csv")
v001_deck_path = os.path.join(REPO_ROOT, "submissions", "v001_random", "deck.csv")

v002_deck = load_deck(v002_deck_path)
v001_deck = load_deck(v001_deck_path)

print(f"v002 deck: {len(v002_deck)} cards loaded from {v002_deck_path}")
print(f"v001 deck: {len(v001_deck)} cards loaded from {v001_deck_path}")
print()

# ---------------------------------------------------------------------------
# Step 0: Deck legality check
# ---------------------------------------------------------------------------
print("=== Step 0: Deck legality check ===")
obs_dict, start_data = battle_start(v002_deck, v001_deck)
if obs_dict is None:
    print(f"FAIL: battle_start returned None obs_dict!")
    print(f"  errorPlayer = {start_data.errorPlayer}")
    print(f"  errorType   = {start_data.errorType}")
    try:
        battle_finish()
    except Exception:
        pass
    sys.exit(1)
else:
    print(f"PASS: battle_start returned valid obs_dict (errorPlayer={start_data.errorPlayer})")
    # Finish this test battle
    try:
        battle_finish()
    except Exception:
        pass

print()

# ---------------------------------------------------------------------------
# Step 1: Load agents and run 20-game smoke test
# ---------------------------------------------------------------------------
print("=== Step 1: 20-game smoke test (v002_lucario vs v001_random) ===")

import importlib.util

def load_agent(submission_dir: str, module_name: str):
    main_path = os.path.join(submission_dir, "main.py")
    original_cwd = os.getcwd()
    try:
        os.chdir(submission_dir)
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod.agent

v002_dir = os.path.join(REPO_ROOT, "submissions", "v002_lucario")
v001_dir = os.path.join(REPO_ROOT, "submissions", "v001_random")

print(f"Loading v002 agent from {v002_dir} ...")
v002_agent = load_agent(v002_dir, "cand_main")
print(f"Loading v001 agent from {v001_dir} ...")
v001_agent = load_agent(v001_dir, "champ_main")
print()

exception_count = 0

def wrapped_v002(obs_dict):
    global exception_count
    try:
        return v002_agent(obs_dict)
    except Exception as e:
        exception_count += 1
        print(f"  EXCEPTION in v002 agent: {e}")
        traceback.print_exc()
        raise

results = tournament(
    wrapped_v002,
    v001_agent,
    v002_deck,
    v001_deck,
    n=20,
    seed=42,
)

print(f"Results (20 games):")
print(f"  v002 wins  : {results['wins_A']}")
print(f"  v001 wins  : {results['wins_B']}")
print(f"  Draws      : {results['draws']}")
print(f"  Winrate v002: {results['winrate_A']:.3f}  (Wilson 95% CI: [{results['wilson_lo']:.3f}, {results['wilson_hi']:.3f}])")
print(f"  Mean move time: {results['mean_move_time_s']*1000:.3f} ms")
print()

if exception_count == 0:
    print(f"PASS: ZERO exceptions across all 20 games.")
else:
    print(f"FAIL: {exception_count} exception(s) detected!")
    sys.exit(1)

print()
print("=== All P1-2 validation steps passed. ===")
