"""v046_pimc_supporter_gate.py — local A/B gate: v046_pimc_supporter vs v046_judgefix (baseline).

K5 experiment (PLAN.md sec K, forum topic 721338 / Chrismaghuhn): replace the
hand-tuned JUDGE/EXPLORER/LILLIE/BOSS supporter-choice priority numbers in
score_play() with a short self-play PIMC rollout whenever >=2 different
supporter cards are legally playable (see submissions/v046_pimc_supporter/main.py).

Methodology (per reference_forum_recon_2026_07_19_alakazam.md / david valor
eval-discipline postmortem): NOT a head-to-head WR between the two variants
(different agents playing each other doesn't isolate the effect) -- instead
each variant plays the SAME three opponents (v017_alakazam_heur is the
primary target matchup; v009a_crustle/v018_grimmsnarl check for regressions
elsewhere) and we compare WR-vs-opponent between variants. N=200/cell first
pass, N>=400 revalidation for anything that looks like a real delta.

Usage:
    .venv/bin/python scripts/v046_pimc_supporter_gate.py --games 200 --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MAX_MOVES_PER_GAME = 2000

BASELINE_SUB = "v046_judgefix"
CANDIDATE_SUB = "v046_pimc_supporter"


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def ztest(w1, n1, w2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = w1 / n1, w2 / n2
    p_pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, float("nan")
    z = (p1 - p2) / se
    from math import erf, sqrt
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p1 - p2, pval


def _load_submission_module(sub_dir_name: str, tag: str):
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_v046pimc_gate_{sub_dir_name}_{tag}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _play_one_game(task: dict) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class
    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)

    player_sub = task["player_sub"]
    opp_sub = task["opp_sub"]
    our_first = task["our_first"]
    seed = task["seed"]
    random.seed(seed)

    player_mod = _load_submission_module(player_sub, "player")
    opp_mod = _load_submission_module(opp_sub, "opp")
    player_deck = list(getattr(player_mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", player_sub, "deck.csv")))
    opp_deck = list(getattr(opp_mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", opp_sub, "deck.csv")))

    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = opp_mod.agent, player_mod.agent

    t_start = time.time()
    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(cell=task["cell"], outcome="crash", moves=0, wall=0.0)

    move_count = 0
    winner = None
    n_exc = 0
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
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    wall = time.time() - t_start

    our_side = 0 if our_first else 1
    if n_exc > 0 or winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    result = dict(cell=task["cell"], outcome=outcome, moves=move_count, wall=wall, n_exc=n_exc)

    # PIMC-specific instrumentation: only the candidate module has _pimc_stats.
    pimc_stats = getattr(player_mod, "_pimc_stats", None)
    if pimc_stats is not None:
        result["pimc_stats"] = dict(pimc_stats)

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--opponents", type=str, default="v017_alakazam_heur,v009a_crustle,v018_grimmsnarl")
    args = ap.parse_args()

    opponents = [o for o in args.opponents.split(",") if o]
    cells = []
    for opp in opponents:
        cells.append((f"baseline_vs_{opp}", BASELINE_SUB, opp))
        cells.append((f"pimc_vs_{opp}", CANDIDATE_SUB, opp))

    ctx = multiprocessing.get_context("spawn")
    queue = []
    for cell_name, player_sub, opp_sub in cells:
        for i in range(args.games):
            queue.append(dict(cell=cell_name, player_sub=player_sub, opp_sub=opp_sub,
                               our_first=(i % 2 == 0),
                               seed=args.seed + (hash((cell_name, i)) % 1_000_000)))
    random.Random(args.seed).shuffle(queue)

    print(f"[gate] {len(queue)} games total ({args.games}/cell x {len(cells)} cells), workers={args.workers}", flush=True)
    t0 = time.time()
    results: dict[str, list[dict]] = {c[0]: [] for c in cells}

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_play_one_game, t): t for t in queue}
        n_done = 0
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                t = futures.pop(f)
                try:
                    res = f.result()
                except Exception:
                    res = dict(cell=t["cell"], outcome="crash", moves=0, wall=0.0, n_exc=1)
                results[res["cell"]].append(res)
                n_done += 1
                if n_done % 40 == 0:
                    print(f"  ... {n_done}/{len(queue)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== RESULTS ({time.time()-t0:.0f}s total) ===")
    summary = {}
    for cell_name, player_sub, opp_sub in cells:
        rows = results[cell_name]
        wins = sum(1 for r in rows if r["outcome"] == "our_win")
        losses = sum(1 for r in rows if r["outcome"] == "opp_win")
        draws = sum(1 for r in rows if r["outcome"] == "draw")
        aborted = sum(1 for r in rows if r["outcome"] in ("aborted", "crash"))
        total_exc = sum(r.get("n_exc", 0) for r in rows)
        n = wins + losses
        wr = wins / n if n else float("nan")
        lo, hi = wilson_ci(wins, n)
        avg_wall = sum(r.get("wall", 0.0) for r in rows) / len(rows) if rows else float("nan")
        max_wall = max((r.get("wall", 0.0) for r in rows), default=float("nan"))
        summary[cell_name] = dict(wins=wins, losses=losses, draws=draws, aborted=aborted, n=n,
                                   wr=wr, lo=lo, hi=hi, total_exc=total_exc,
                                   avg_wall=avg_wall, max_wall=max_wall)
        print(f"{cell_name:20s} ({player_sub} vs {opp_sub}): W={wins} L={losses} D={draws} "
              f"aborted={aborted} agent_exc={total_exc}  WR={wr:.3f}  Wilson95=[{lo:.3f},{hi:.3f}]  "
              f"avg_game_wall={avg_wall:.2f}s max={max_wall:.2f}s")

    print("\n=== v046_pimc_supporter vs v046_judgefix delta (per opponent) ===")
    for opp in opponents:
        a, b = f"pimc_vs_{opp}", f"baseline_vs_{opp}"
        sa, sb = summary[a], summary[b]
        if sa["n"] and sb["n"]:
            diff, pval = ztest(sa["wins"], sa["n"], sb["wins"], sb["n"])
            print(f"{opp:24s}: pimc WR={sa['wr']:.3f} [{sa['lo']:.3f},{sa['hi']:.3f}]  vs  "
                  f"baseline WR={sb['wr']:.3f} [{sb['lo']:.3f},{sb['hi']:.3f}]  "
                  f"delta={diff:+.3f}  z-test p={pval:.4f}")

    # PIMC-specific instrumentation aggregated across all candidate-side games.
    print("\n=== PIMC supporter-choice instrumentation (candidate cells only) ===")
    pimc_calls = pimc_used = pimc_no_signal = pimc_exc = pimc_rollouts = 0
    pimc_time = 0.0
    n_candidate_games = 0
    for cell_name, player_sub, opp_sub in cells:
        if player_sub != CANDIDATE_SUB:
            continue
        for r in results[cell_name]:
            ps = r.get("pimc_stats")
            if not ps:
                continue
            n_candidate_games += 1
            pimc_calls += ps.get("calls", 0)
            pimc_used += ps.get("decisions_used", 0)
            pimc_no_signal += ps.get("decisions_no_signal", 0)
            pimc_exc += ps.get("exceptions", 0)
            pimc_rollouts += ps.get("rollouts_total", 0)
            pimc_time += ps.get("time_total", 0.0)
    print(f"games with instrumentation: {n_candidate_games}")
    print(f"total PIMC decision points: {pimc_calls}  (used={pimc_used}, no_signal_fallback={pimc_no_signal}, "
          f"internal_exceptions={pimc_exc})")
    print(f"total rollouts completed: {pimc_rollouts}  total PIMC wall time: {pimc_time:.1f}s")
    if pimc_calls:
        print(f"avg PIMC decision points/game: {pimc_calls/max(n_candidate_games,1):.2f}  "
              f"avg time/decision: {pimc_time/pimc_calls:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
