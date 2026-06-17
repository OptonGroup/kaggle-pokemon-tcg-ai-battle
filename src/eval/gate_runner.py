"""Gate runner — pits a candidate submission against the champion.

Usage:
    python src/eval/gate_runner.py <candidate_version> [champion_version] [n_games] [seed]

Example:
    python src/eval/gate_runner.py v002_lucario v001_random 300 42
    python src/eval/gate_runner.py v002_lucario  # uses champion.txt, 50 games

The champion defaults to the contents of results/champion.txt.
n_games defaults to 50.
seed defaults to 42.

Submission Gate (from PLAN.md Guardrails):
    PASS iff ALL of:
        1. winrate_A >= 0.55
        2. n_games >= 300
        3. wilson_lo > 0.50
    else FAIL.
"""
import sys
import os
import importlib.util

# Ensure src/ is on the path so `import cg` and sim_harness resolve
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)

from sim_harness import tournament


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_agent(submission_dir: str, module_name: str, return_module: bool = False):
    """Import main.py from a submission directory as a uniquely-named module.

    Before importing, os.chdir() into the submission directory so any
    module-level code that reads 'deck.csv' from CWD finds the right file.
    CWD is restored afterwards.

    If return_module=True, returns (agent_fn, module); else returns agent_fn.
    """
    main_path = os.path.join(submission_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"No main.py found in {submission_dir}")

    original_cwd = os.getcwd()
    try:
        os.chdir(submission_dir)
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)

    if not hasattr(mod, "agent"):
        raise AttributeError(f"No 'agent' function found in {main_path}")
    if return_module:
        return mod.agent, mod
    return mod.agent


def _load_deck(submission_dir: str) -> list[int]:
    """Read deck.csv from a submission directory."""
    deck_path = os.path.join(submission_dir, "deck.csv")
    if not os.path.exists(deck_path):
        raise FileNotFoundError(f"No deck.csv found in {submission_dir}")
    with open(deck_path) as f:
        cards = [int(line.strip()) for line in f if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck.csv in {submission_dir} has {len(cards)} cards (expected 60)")
    return cards


def _read_champion_txt(repo_root: str) -> str:
    """Read the champion version from results/champion.txt."""
    champ_path = os.path.join(repo_root, "results", "champion.txt")
    if not os.path.exists(champ_path):
        raise FileNotFoundError(f"results/champion.txt not found at {champ_path}")
    with open(champ_path) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse CLI args
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python src/eval/gate_runner.py <candidate_version> [champion_version] [n_games] [seed]")
        sys.exit(1)

    candidate_version = args[0]
    # args[1] is champion version OR an int (n_games) if only 2 positional args given
    champion_version = None
    n_games = 50
    seed = 42

    # Flexible parsing:
    #   2-arg: <candidate> <n_games>                  → champion from txt
    #   3-arg: <candidate> <champion> <n_games>        → if args[1] is not int
    #          <candidate> <n_games> <seed>            → if args[1] is int
    #   4-arg: <candidate> <champion> <n_games> <seed>
    if len(args) >= 2:
        try:
            n_games = int(args[1])
            # args[1] is n_games; champion from txt; args[2] is seed if present
            if len(args) >= 3:
                try:
                    seed = int(args[2])
                except ValueError:
                    pass
        except ValueError:
            # args[1] is champion version
            champion_version = args[1]
            if len(args) >= 3:
                try:
                    n_games = int(args[2])
                except ValueError:
                    pass
            if len(args) >= 4:
                try:
                    seed = int(args[3])
                except ValueError:
                    pass

    # Determine repo root (two levels up from this file: src/eval/ -> src/ -> repo/)
    repo_root = os.path.dirname(_SRC_DIR)

    if champion_version is None:
        champion_version = _read_champion_txt(repo_root)

    cand_dir = os.path.join(repo_root, "submissions", candidate_version)
    champ_dir = os.path.join(repo_root, "submissions", champion_version)

    for label, d in [("Candidate", cand_dir), ("Champion", champ_dir)]:
        if not os.path.isdir(d):
            print(f"ERROR: {label} directory not found: {d}")
            sys.exit(1)

    print(f"=== Submission Gate ===")
    print(f"  Candidate : {candidate_version}  ({cand_dir})")
    print(f"  Champion  : {champion_version}  ({champ_dir})")
    print(f"  n_games   : {n_games}")
    print(f"  seed      : {seed}")
    print()

    # Load decks explicitly
    cand_deck = _load_deck(cand_dir)
    champ_deck = _load_deck(champ_dir)
    print(f"Decks loaded: candidate={len(cand_deck)} cards, champion={len(champ_deck)} cards")

    # Load agents with unique module names and CWD trick for deck.csv
    print(f"Loading candidate agent from {cand_dir}/main.py ...")
    candidate_agent, cand_mod = _load_agent(cand_dir, "cand_main", return_module=True)
    print(f"Loading champion agent from {champ_dir}/main.py ...")
    champion_agent = _load_agent(champ_dir, "champ_main")
    print()

    print(f"Running {n_games} games (candidate=A, champion=B) ...")
    results = tournament(
        candidate_agent,
        champion_agent,
        cand_deck,
        champ_deck,
        n=n_games,
        seed=seed,
    )

    wins_A = results["wins_A"]
    wins_B = results["wins_B"]
    draws = results["draws"]
    winrate_A = results["winrate_A"]
    lo = results["wilson_lo"]
    hi = results["wilson_hi"]
    mean_time_ms = results["mean_move_time_s"] * 1000

    print()
    print("=== Results ===")
    print(f"  Candidate wins : {wins_A}")
    print(f"  Champion wins  : {wins_B}")
    print(f"  Draws          : {draws}")
    print(f"  Total games    : {n_games}")
    print(f"  Winrate (excl. draws): {winrate_A:.4f}")
    print(f"  Wilson 95% CI        : [{lo:.4f}, {hi:.4f}]")
    print(f"  Mean move time       : {mean_time_ms:.3f} ms")

    # Instrumentation counters (present in IL agent and hybrid agent)
    n_model = getattr(cand_mod, "_n_model_decisions", None)
    n_heur = getattr(cand_mod, "_n_heuristic_fallback", None)
    n_trivial = getattr(cand_mod, "_n_trivial_fallback", None)
    # v005_hybrid: _n_heuristic_fallback = IL_override count (not a true fallback)
    lam = getattr(cand_mod, "_LAMBDA", None)
    if n_model is not None:
        total_dec = (n_model or 0) + (n_trivial or 0)
        pct_model = 100.0 * (n_model or 0) / total_dec if total_dec > 0 else 0.0
        il_override = n_heur or 0
        pct_override = 100.0 * il_override / (n_model or 1) if n_model else 0.0
        print()
        print("=== Candidate Decision Breakdown ===")
        if lam is not None:
            print(f"  HYBRID_LAMBDA      : {lam}")
        print(f"  Model decisions    : {n_model}  ({pct_model:.1f}%)")
        if lam is not None:
            print(f"  IL overrode top-1  : {il_override}  ({pct_override:.1f}% of model decisions)")
        else:
            print(f"  Heuristic fallback : {n_heur}  ({100.0*(n_heur or 0)/total_dec if total_dec else 0:.1f}%)")
        print(f"  Exception fallback : {n_trivial}  ({100.0*(n_trivial or 0)/total_dec if total_dec else 0:.1f}%)")
        print(f"  Total decisions    : {total_dec}")
    print()

    # Gate thresholds (from PLAN.md Guardrails)
    WINRATE_THRESHOLD = 0.55
    N_GAMES_THRESHOLD = 300
    WILSON_LO_THRESHOLD = 0.50

    pass_winrate = winrate_A >= WINRATE_THRESHOLD
    pass_n_games = n_games >= N_GAMES_THRESHOLD
    pass_wilson = lo > WILSON_LO_THRESHOLD

    all_pass = pass_winrate and pass_n_games and pass_wilson

    print("=== Gate Verdict ===")
    print(f"  winrate_A >= {WINRATE_THRESHOLD}     : {'PASS' if pass_winrate else 'FAIL'}  ({winrate_A:.4f})")
    print(f"  n_games >= {N_GAMES_THRESHOLD}       : {'PASS' if pass_n_games else 'FAIL'}  ({n_games})")
    print(f"  wilson_lo > {WILSON_LO_THRESHOLD}     : {'PASS' if pass_wilson else 'FAIL'}  ({lo:.4f})")
    print()
    if all_pass:
        print("VERDICT: PASS — candidate clears all gate conditions.")
    else:
        failed = []
        if not pass_winrate:
            failed.append(f"winrate_A={winrate_A:.4f} < {WINRATE_THRESHOLD}")
        if not pass_n_games:
            failed.append(f"n_games={n_games} < {N_GAMES_THRESHOLD}")
        if not pass_wilson:
            failed.append(f"wilson_lo={lo:.4f} <= {WILSON_LO_THRESHOLD}")
        print(f"VERDICT: FAIL — failed condition(s): {'; '.join(failed)}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
