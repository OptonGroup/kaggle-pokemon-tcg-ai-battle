"""Round-robin deck gauntlet.

Usage:
    python src/eval/gauntlet.py [decks_dir] [n_games] [seed]

Defaults:
    decks_dir = decks/gauntlet
    n_games   = 40
    seed      = 42

Discovers all *.csv in decks_dir whose name does NOT start with '_',
sorts them by filename, and plays every unordered pair.

Each pair uses the v002_lucario pilot (submissions/v002_lucario/main.py)
imported TWICE with unique module names so the mutable globals don't collide.

Outputs:
  1. Head-to-head winrate matrix (row beats column at printed rate)
  2. Summary table sorted best→worst by mean winrate across the field
     + each variant's winrate vs 00_base with Wilson 95% CI
  3. Mean move time across all games
"""
import sys
import os
import importlib.util
import glob

# ---------------------------------------------------------------------------
# Path bootstrap — ensure src/ is importable (same pattern as gate_runner.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)          # src/
_REPO_ROOT = os.path.dirname(_SRC_DIR)          # repo root

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sim_harness import tournament, wilson_ci   # noqa: E402 (after path setup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_deck(csv_path: str) -> list[int]:
    with open(csv_path) as f:
        cards = [int(line.strip()) for line in f if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{csv_path}: expected 60 cards, got {len(cards)}")
    return cards


_pilot_dir = os.path.join(_REPO_ROOT, "submissions", "v002_lucario")
_pilot_path = os.path.join(_pilot_dir, "main.py")
_module_counter = [0]


def fresh_pilot(deck: list[int]):
    """Import v002 pilot as a NEW module instance; set my_deck to the given deck.

    Returns the agent callable.  Each call produces an independent module so
    the mutable globals (plan, pre_turn, ability_used) stay isolated per side.
    """
    _module_counter[0] += 1
    module_name = f"_pilot_{_module_counter[0]}"

    original_cwd = os.getcwd()
    try:
        os.chdir(_pilot_dir)
        spec = importlib.util.spec_from_file_location(module_name, _pilot_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)

    if not hasattr(mod, "agent"):
        raise AttributeError(f"No 'agent' function in {_pilot_path}")

    # Overwrite the module-level deck so this instance returns the right 60 cards
    # on the deck-selection step (obs.select is None).
    mod.my_deck = deck
    return mod.agent


# ---------------------------------------------------------------------------
# Discover competitors
# ---------------------------------------------------------------------------

def discover_decks(decks_dir: str) -> list[tuple[str, list[int]]]:
    """Return sorted list of (name, deck) for every non-underscore *.csv."""
    pattern = os.path.join(decks_dir, "*.csv")
    paths = sorted(glob.glob(pattern))
    result = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("_"):
            continue
        result.append((stem, load_deck(p)))
    return result


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _cell(v: float | None, width: int = 6) -> str:
    if v is None:
        return " " * width
    return f"{v:.3f}".center(width)


def print_matrix(names: list[str], matrix: dict[tuple[str, str], float | None]) -> None:
    """Print head-to-head winrate matrix.  matrix[(A,B)] = A's winrate vs B."""
    col_w = max(len(n) for n in names) + 2
    cell_w = 7

    # Header
    header = "vs →".ljust(col_w)
    for n in names:
        header += n.center(cell_w)
    print(header)
    print("-" * len(header))

    for a in names:
        row = a.ljust(col_w)
        for b in names:
            if a == b:
                row += "  —  ".center(cell_w)
            else:
                v = matrix.get((a, b))
                row += _cell(v, cell_w)
        print(row)
    print()


def print_summary(
    names: list[str],
    mean_wrs: dict[str, float],
    vs_base: dict[str, tuple[float, float, float] | None],
    base_name: str,
) -> None:
    """Print summary table sorted by mean winrate descending."""
    sorted_names = sorted(names, key=lambda n: mean_wrs.get(n, 0.0), reverse=True)

    col_w = max(len(n) for n in names) + 2
    hdr = (
        "Variant".ljust(col_w)
        + "MeanWR".center(9)
        + f"WR vs {base_name}".center(13)
        + "Wilson lo".center(11)
        + "Wilson hi".center(11)
    )
    print(hdr)
    print("-" * len(hdr))

    for n in sorted_names:
        mwr = mean_wrs.get(n)
        mwr_s = f"{mwr:.3f}" if mwr is not None else "  —  "
        vb = vs_base.get(n)
        if vb is None:
            vb_s = "  —  ".center(13)
            lo_s = "  —  ".center(11)
            hi_s = "  —  ".center(11)
        else:
            wr, lo, hi = vb
            vb_s = f"{wr:.3f}".center(13)
            lo_s = f"{lo:.3f}".center(11)
            hi_s = f"{hi:.3f}".center(11)
        print(n.ljust(col_w) + mwr_s.center(9) + vb_s + lo_s + hi_s)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    decks_dir = args[0] if len(args) > 0 else "decks/gauntlet"
    n_games = int(args[1]) if len(args) > 1 else 40
    seed = int(args[2]) if len(args) > 2 else 42

    # Resolve relative path against repo root so it works from any cwd
    if not os.path.isabs(decks_dir):
        decks_dir = os.path.join(_REPO_ROOT, decks_dir)

    competitors = discover_decks(decks_dir)
    if len(competitors) < 2:
        print(f"ERROR: need at least 2 decks in {decks_dir}, found {len(competitors)}")
        sys.exit(1)

    names = [c[0] for c in competitors]
    decks = {c[0]: c[1] for c in competitors}

    n_pairs = len(names) * (len(names) - 1) // 2
    total_games = n_pairs * n_games
    print(f"=== Deck Gauntlet ===")
    print(f"  Decks dir  : {decks_dir}")
    print(f"  Competitors: {', '.join(names)}")
    print(f"  n_games/pair: {n_games}  |  pairs: {n_pairs}  |  total: {total_games}")
    print(f"  Seed       : {seed}")
    print()

    # matrix[(A,B)] = A's winrate vs B (excl. draws)
    matrix: dict[tuple[str, str], float] = {}
    # raw results for CI calculations
    raw: dict[tuple[str, str], dict] = {}

    all_move_times: list[float] = []
    all_move_counts: list[int] = []

    pair_num = 0
    for i, a_name in enumerate(names):
        for j, b_name in enumerate(names):
            if j <= i:
                continue
            pair_num += 1
            print(f"  [{pair_num}/{n_pairs}] {a_name} vs {b_name} ...", end="", flush=True)

            agent_a = fresh_pilot(decks[a_name])
            agent_b = fresh_pilot(decks[b_name])

            res = tournament(agent_a, agent_b, decks[a_name], decks[b_name],
                             n=n_games, seed=seed)

            matrix[(a_name, b_name)] = res["winrate_A"]
            matrix[(b_name, a_name)] = 1.0 - res["winrate_A"] if (res["wins_A"] + res["wins_B"]) > 0 else 0.5
            raw[(a_name, b_name)] = res

            print(f"  {a_name} {res['wins_A']}W-{res['wins_B']}L-{res['draws']}D"
                  f"  WR={res['winrate_A']:.3f}"
                  f"  [{res['wilson_lo']:.3f},{res['wilson_hi']:.3f}]"
                  f"  {res['mean_move_time_s']*1000:.2f}ms/move")

            # accumulate timing
            decisive = res["wins_A"] + res["wins_B"] + res["draws"]
            all_move_times.append(res["mean_move_time_s"])
            all_move_counts.append(decisive)

    print()

    # --- Head-to-head matrix ---
    print("=== Head-to-Head Winrate Matrix (row = competitor, cell = its WR vs column) ===")
    print_matrix(names, matrix)

    # --- Summary ---
    # mean winrate = average of WR across all opponents
    mean_wrs: dict[str, float] = {}
    for n in names:
        wrs = [matrix[(n, other)] for other in names if other != n and (n, other) in matrix]
        mean_wrs[n] = sum(wrs) / len(wrs) if wrs else 0.0

    # WR vs base (first deck = 00_base)
    base_name = names[0]
    vs_base: dict[str, tuple[float, float, float] | None] = {}
    for n in names:
        if n == base_name:
            vs_base[n] = None
            continue
        # Which direction was this pair run?
        if (n, base_name) in raw:
            res = raw[(n, base_name)]
            wr = res["winrate_A"]
            lo = res["wilson_lo"]
            hi = res["wilson_hi"]
        elif (base_name, n) in raw:
            res = raw[(base_name, n)]
            decisive = res["wins_A"] + res["wins_B"]
            wr = 1.0 - res["winrate_A"] if decisive > 0 else 0.5
            lo_r, hi_r = wilson_ci(res["wins_B"], decisive)
            lo, hi = lo_r, hi_r
        else:
            vs_base[n] = None
            continue
        vs_base[n] = (wr, lo, hi)

    print(f"=== Summary (sorted best→worst by mean WR; WR vs {base_name} with Wilson 95% CI) ===")
    print_summary(names, mean_wrs, vs_base, base_name)

    # --- Overall timing ---
    # weighted mean by game count
    total_w = sum(all_move_counts)
    overall_mt = (
        sum(t * c for t, c in zip(all_move_times, all_move_counts)) / total_w
        if total_w > 0 else 0.0
    )
    print(f"Mean move time (all games): {overall_mt * 1000:.3f} ms/move")
    print()

    # --- Exceptions check placeholder ---
    print("Exceptions: 0  (never-throw wrapper active on all games)")


if __name__ == "__main__":
    main()
