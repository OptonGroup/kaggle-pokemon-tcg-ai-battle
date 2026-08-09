"""Expanded gate for v033_plan_d_ppo.

Plays v033 (pure-NN PPO policy, non-god-mode) vs an opponent, side-alternating.
Both agents see ONLY the cabt obs (cg_obs) — the first honest measurement.

Features over d3_gate.py:
  - --agent-dir <path>: override our agent's submission dir (default:
    submissions/v033_plan_d_ppo). Relative paths resolved against repo root.
  - --opp-deck <path>: override opponent's deck.csv (for cross-deck robustness
    tests vs meta archetypes with v009a pilot). Also overrides opp module's
    `my_deck` global if present, so deck-selection step returns the alt deck.
  - --out <path>: append result to a JSON state file (results/gates/...).
  - --tag <name>: label for the JSON record.
  - --opponent, --games, --seed: same as d3_gate.

Reports:
  - H2H WR + Wilson 95% CI from v033's perspective.
  - v033 counters: nn_decisions / fallback / trivial / exceptions.
  - v033 never-throw: any exception in v033 path = RED FLAG (converter/NN
    breaking on unfamiliar cards in cross-deck).

Usage:
    # vs v009a to 300 games
    python scripts/gate_expanded.py --opponent v009a_crustle --games 300 \
        --tag v033_vs_v009a_300 --out results/gates/v033_gate_expanded.json

    # cross-deck: v033 (Crustle) vs v009a-pilot with meta deck
    python scripts/gate_expanded.py --opponent v009a_crustle --games 80 \
        --opp-deck data/deck_scan/meta_decks/alakazam.csv \
        --tag v033_vs_alakazam_v009apilot --out results/gates/v033_gate_expanded.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# cg alias shim BEFORE loading any agent that does `from cg.api import ...`.
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

V033_DIR = os.path.join(ROOT, "submissions", "v033_plan_d_ppo")
MAX_MOVES = 2000


def wilson(wins, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / den
    return (max(0.0, c - m), min(1.0, c + m))


def load_agent(sub_dir, tag):
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_gate_{tag}_{id(sub_dir)}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"no agent() in {main_path}")
    return mod


def load_deck_file(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def load_deck(sub_dir):
    return load_deck_file(os.path.join(sub_dir, "deck.csv"))


def override_opp_deck(opp_mod, alt_deck, opp_dir):
    """Replace opp module's deck so deck-selection step returns the alt deck."""
    if hasattr(opp_mod, "my_deck"):
        opp_mod.my_deck = list(alt_deck)
    # some agents keep a _deck or DECK global
    for attr in ("_deck", "DECK", "deck"):
        if hasattr(opp_mod, attr):
            try:
                setattr(opp_mod, attr, list(alt_deck))
            except Exception:
                pass


def play_game(v033_mod, opp_mod, v033_is_left, opp_deck_override=None, forensic=None):
    """Play one game. v033_is_left: True if v033 controls deck0/left side.
    opp_deck_override: list of 60 card IDs to use as opponent's deck in battle_start.
    Returns (v033_outcome in {+1,-1,0}, n_moves, exc_side)."""
    left_deck = load_deck(V033_DIR)
    opp_deck = opp_deck_override if opp_deck_override is not None else load_deck(os.path.dirname(opp_mod.__file__))
    if v033_is_left:
        deck0, deck1 = left_deck, opp_deck
        agent0, agent1 = v033_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, left_deck
        agent0, agent1 = opp_mod.agent, v033_mod.agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return 0.0, 0, "load"
    n_moves = 0
    exc_side = None
    winner = None
    forensic_count = 0
    while n_moves < MAX_MOVES:
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            winner = obs.current.result
            break
        your = obs.current.yourIndex if obs.current else 0
        agent_fn = agent0 if your == 0 else agent1
        sel = obs_dict.get("select")
        is_v033_turn = (your == 0 and v033_is_left) or (your == 1 and not v033_is_left)
        try:
            choice = agent_fn(obs_dict)
        except Exception:
            exc_side = "v033" if is_v033_turn else "opp"
            winner = 1 - your
            break
        # legality
        opts = sel.get("option") or [] if sel else []
        if sel is not None:
            n_opt = len(opts)
            mn = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
            mx = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)
            ok = (isinstance(choice, list) and all(isinstance(i, int) and 0 <= i < n_opt for i in choice)
                  and mn <= len(choice) <= mx and len(set(choice)) == len(choice))
            if not ok:
                exc_side = "v033" if is_v033_turn else "opp"
                winner = 1 - your
                break
        if forensic is not None and is_v033_turn and forensic_count < 3:
            forensic.append((n_moves, your, len(opts), (choice[:3] if choice else [])))
            forensic_count += 1
        try:
            obs_dict = battle_select(choice)
        except Exception:
            exc_side = "v033" if is_v033_turn else "opp"
            winner = 1 - your
            break
        n_moves += 1
    try:
        battle_finish()
    except Exception:
        pass
    if winner is None:
        return 0.0, n_moves, exc_side  # aborted -> draw-ish
    if winner == 2:
        return 0.0, n_moves, exc_side
    v033_side = 0 if v033_is_left else 1
    out = 1.0 if winner == v033_side else -1.0
    return out, n_moves, exc_side


def run_match(v033_mod, opp_mod, n_games, opp_deck_override=None, seed=20260716):
    outcomes = []
    moves_list = []
    exc_v033 = 0
    exc_opp = 0
    forensic = None
    t0 = time.perf_counter()
    for g in range(n_games):
        v033_is_left = (g % 2 == 0)
        forensic = [] if g == 0 else None
        out, nmoves, exc = play_game(v033_mod, opp_mod, v033_is_left,
                                     opp_deck_override=opp_deck_override, forensic=forensic)
        outcomes.append(out)
        moves_list.append(nmoves)
        if exc == "v033":
            exc_v033 += 1
        elif exc == "opp":
            exc_opp += 1
        if g == 0 and forensic is not None:
            print("[forensic] game0 v033 decisions (move, yourIndex, n_opts, choice[:3]):")
            for f in forensic:
                print(f"  move={f[0]:3d} your={f[1]} n_opts={f[2]} choice[:3]={f[3]}")
    elapsed = time.perf_counter() - t0
    return {
        "outcomes": outcomes,
        "moves_list": moves_list,
        "exc_v033": exc_v033,
        "exc_opp": exc_opp,
        "elapsed": elapsed,
        "forensic_game0": forensic if (n_games > 0 and forensic is not None) else None,
    }


def summarize(result, opponent, tag, opp_deck_override_name, threshold_pass=0.65):
    outcomes = result["outcomes"]
    moves_list = result["moves_list"]
    wins = sum(1 for o in outcomes if o > 0)
    losses = sum(1 for o in outcomes if o < 0)
    draws = sum(1 for o in outcomes if o == 0)
    decisive = wins + losses
    we = wins + 0.5 * draws
    n = len(outcomes)
    wr = we / n if n else float("nan")
    lo, hi = wilson(we, n)
    avg_moves = sum(moves_list) / max(1, len(moves_list))
    summary = {
        "tag": tag,
        "opponent": opponent,
        "opp_deck_override": opp_deck_override_name,
        "n_games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "WR": wr,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "decisive_WR": wins / max(decisive, 1),
        "decisive_n": decisive,
        "avg_moves": avg_moves,
        "min_moves": min(moves_list) if moves_list else 0,
        "max_moves": max(moves_list) if moves_list else 0,
        "exc_v033": result["exc_v033"],
        "exc_opp": result["exc_opp"],
        "elapsed_s": result["elapsed"],
    }
    return summary, wr, lo, hi, wins, losses, draws, avg_moves


def main():
    global V033_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", type=str, default=V033_DIR,
                    help="path to our agent's submission dir "
                         "(default: submissions/v033_plan_d_ppo; relative paths "
                         "resolved against repo root)")
    ap.add_argument("--opponent", type=str, default="v009a_crustle")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--opp-deck", type=str, default=None,
                    help="path to alt deck.csv (cross-deck test)")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="path to JSON state file (append)")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="WR threshold for PASS (vs non-random)")
    args = ap.parse_args()

    V033_DIR = args.agent_dir if os.path.isabs(args.agent_dir) else os.path.join(ROOT, args.agent_dir)

    opp_dir = os.path.join(ROOT, "submissions", args.opponent)
    if not os.path.isdir(opp_dir):
        print(f"opponent dir not found: {opp_dir}", file=sys.stderr)
        return 2

    opp_deck_override = None
    opp_deck_override_name = None
    if args.opp_deck:
        opp_deck_override_name = os.path.relpath(args.opp_deck, ROOT)
        opp_deck_override = load_deck_file(args.opp_deck)
        assert len(opp_deck_override) == 60, f"alt deck has {len(opp_deck_override)} cards"

    tag = args.tag or ("v033_vs_" + args.opponent +
                       ("_" + os.path.basename(args.opp_deck).replace(".csv", "") if args.opp_deck else ""))

    print(f"[gate] v033 vs {args.opponent}  games={args.games}  (native macOS, non-god-mode)")
    if opp_deck_override is not None:
        print(f"[gate] opp deck overridden: {opp_deck_override_name}")
    v033 = load_agent(V033_DIR, "v033")
    opp = load_agent(opp_dir, "opp")
    if opp_deck_override is not None:
        override_opp_deck(opp, opp_deck_override, opp_dir)
        print(f"[gate] opp.my_deck overridden (len={len(getattr(opp,'my_deck',[]))})")
    print(f"[gate] v033 torch_ok={v033._torch_ok}  model_loaded={v033._model is not None}")

    result = run_match(v033, opp, args.games, opp_deck_override=opp_deck_override, seed=args.seed)

    summ, wr, lo, hi, wins, losses, draws, avg_moves = summarize(
        result, args.opponent, tag, opp_deck_override_name, args.threshold)

    # v033 counters (cumulative across all games, since same module)
    nn_dec = getattr(v033, "_n_nn_decisions", -1)
    fb_dec = getattr(v033, "_n_fallback_decisions", -1)
    triv = getattr(v033, "_n_trivial_fallback", -1)
    vexc = getattr(v033, "_n_exceptions", -1)
    summ["v033_nn_decisions"] = nn_dec
    summ["v033_fallback_decisions"] = fb_dec
    summ["v033_trivial_fallback"] = triv
    summ["v033_exceptions"] = vexc

    verdict = "PASS" if (wr >= args.threshold and vexc == 0 and result["exc_v033"] == 0) else "FAIL"
    summ["verdict"] = verdict
    summ["threshold"] = args.threshold

    print("=" * 70)
    print(f"v033 vs {args.opponent}" +
          (f" [deck={opp_deck_override_name}]" if opp_deck_override_name else "") +
          f"  (n_games={args.games}, elapsed={result['elapsed']:.1f}s)")
    print("=" * 70)
    print(f"v033: W={wins} L={losses} D={draws}  WR={wr:.4f}  Wilson95=[{lo:.4f},{hi:.4f}]")
    print(f"  decisive WR (excl draws) = {wins/max(wins+losses,1):.4f} over {wins+losses} decisive")
    print(f"  avg moves/game = {avg_moves:.1f}  (range {min(result['moves_list'])}..{max(result['moves_list'])})")
    print(f"  exceptions in-game: v033={result['exc_v033']}  opp={result['exc_opp']}")
    print(f"v033 counters: nn_decisions={nn_dec}  fallback={fb_dec}  trivial={triv}  exceptions={vexc}")
    print(f"  VERDICT (WR>={args.threshold}, v033_exc=0): {verdict}")
    print("=" * 70)

    if args.out:
        out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
        state = {}
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    state = json.load(f)
            except Exception:
                state = {}
        records = state.setdefault("records", [])
        records.append(summ)
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[gate] appended record '{tag}' to {out_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())