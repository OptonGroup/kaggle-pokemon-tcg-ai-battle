"""v046_field_gate.py — v046_final vs v045 baseline across the full local opponent field.

Extends scripts/v046_deckrace_gate.py (which only covered the two matchups the v046
patch directly targeted, Alakazam/Crustle proxies) to the broader field we normally gate
against: real scripted pilots (v001/v009a/v017/v018/v036) plus meta decks played by the
generic greedy-attack pilot (decks/meta/*.csv, same convention as scripts/h3_gate.py's
greedy_meta_agent -- both sides of a meta: cell face an IDENTICAL generic opponent, so a
meta: matchup measures OUR play quality on that deck's board state, not the deck's own
"skill"). Purpose: confirm v046_final's Alakazam gain isn't offset by a regression
somewhere else in the field before considering it for submission.

Usage:
    .venv/bin/python scripts/v046_field_gate.py --games 200 --workers 4
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

PLAYER_VARIANTS = ["v044_archaludon_abomasnow", "v045_archaludon_judge_fix", "v046_final"]

OPPONENTS = [
    "v001_random",
    "v009a_crustle",
    "v017_alakazam_heur",
    "v018_grimmsnarl",
    "v036_archaludon_patched",
    "meta:alakazam_aidy",
    "meta:grimmsnarl_kazuki",
    "meta:cubchoo_mtn",
    "meta:MegaLucario",
    "meta:MegaStarmie",
    "meta:Dragapult",
    "meta:Garchomp",
]


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
        spec = importlib.util.spec_from_file_location(f"_v046_field_{sub_dir_name}_{tag}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


# --- generic greedy opponent for meta: decks (mirrors h3_gate.greedy_meta_agent) ---

_ATTACK_DMG_MAP = None


def _dmg_map():
    global _ATTACK_DMG_MAP
    if _ATTACK_DMG_MAP is None:
        from cg_native.api import all_attack
        _ATTACK_DMG_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DMG_MAP


_OT_YES, _OT_ATTACK = 1, 13
_ST_YES_NO, _ST_COUNT, _SC_MAIN = 9, 8, 0


def greedy_meta_agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return []
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    sel_type = sel.get("type")
    if sel_type == _ST_YES_NO:
        for i in range(n):
            if (opts[i].get("type") if isinstance(opts[i], dict) else getattr(opts[i], "type", None)) == _OT_YES:
                return [i]
        return [0]
    if sel_type == _ST_COUNT:
        best = max(range(n), key=lambda i: (opts[i].get("number") or 0))
        return [best]
    min_c = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
    max_c = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)
    if sel.get("context") == _SC_MAIN:
        atk_idxs = [i for i in range(n) if opts[i].get("type") == _OT_ATTACK]
        if atk_idxs:
            dmap = _dmg_map()
            best = max(atk_idxs, key=lambda i: (dmap.get(opts[i].get("attackId"), 0) or 0))
            return [best]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


def _resolve_opponent(opp_spec: str, tag: str):
    if opp_spec.startswith("meta:"):
        deck_name = opp_spec[5:]
        deck = _load_deck_csv(os.path.join(ROOT, "decks", "meta", f"{deck_name}.csv"))
        return greedy_meta_agent, deck
    mod = _load_submission_module(opp_spec, tag)
    deck = list(getattr(mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", opp_spec, "deck.csv")))
    return mod.agent, deck


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
    opp_spec = task["opp_spec"]
    our_first = task["our_first"]
    seed = task["seed"]
    random.seed(seed)

    player_mod = _load_submission_module(player_sub, "player")
    player_deck = list(getattr(player_mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", player_sub, "deck.csv")))
    opp_agent, opp_deck = _resolve_opponent(opp_spec, "opp")

    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_mod.agent, opp_agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = opp_agent, player_mod.agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(cell=task["cell"], outcome="crash", moves=0)

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

    our_side = 0 if our_first else 1
    if n_exc > 0 or winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return dict(cell=task["cell"], outcome=outcome, moves=move_count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()

    cells = []
    for player_sub in PLAYER_VARIANTS:
        for opp_spec in OPPONENTS:
            cells.append((f"{player_sub}__vs__{opp_spec}", player_sub, opp_spec))

    ctx = multiprocessing.get_context("spawn")
    queue = []
    for cell_name, player_sub, opp_spec in cells:
        for i in range(args.games):
            queue.append(dict(cell=cell_name, player_sub=player_sub, opp_spec=opp_spec,
                               our_first=(i % 2 == 0),
                               seed=args.seed + (hash((cell_name, i)) % 1_000_000)))
    random.Random(args.seed).shuffle(queue)

    print(f"[field-gate] {len(queue)} games total ({args.games}/cell x {len(cells)} cells), "
          f"workers={args.workers}", flush=True)
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
                    res = dict(cell=t["cell"], outcome="crash", moves=0)
                results[res["cell"]].append(res)
                n_done += 1
                if n_done % 100 == 0:
                    print(f"  ... {n_done}/{len(queue)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== RESULTS ({time.time()-t0:.0f}s total) ===")
    summary = {}
    for cell_name, player_sub, opp_spec in cells:
        rows = results[cell_name]
        wins = sum(1 for r in rows if r["outcome"] == "our_win")
        losses = sum(1 for r in rows if r["outcome"] == "opp_win")
        draws = sum(1 for r in rows if r["outcome"] == "draw")
        aborted = sum(1 for r in rows if r["outcome"] in ("aborted", "crash"))
        n = wins + losses
        wr = wins / n if n else float("nan")
        lo, hi = wilson_ci(wins, n)
        summary[cell_name] = dict(wins=wins, losses=losses, draws=draws, aborted=aborted, n=n, wr=wr, lo=lo, hi=hi,
                                   player_sub=player_sub, opp_spec=opp_spec)

    header = "".join(f"{v:>16s}" for v in PLAYER_VARIANTS)
    print(f"\n{'opponent':26s}{header}")
    for opp_spec in OPPONENTS:
        row = "".join(f"{summary[f'{v}__vs__{opp_spec}']['wr']:>16.3f}" for v in PLAYER_VARIANTS)
        print(f"{opp_spec:26s}{row}")

    totals = {v: [0, 0] for v in PLAYER_VARIANTS}
    for opp_spec in OPPONENTS:
        for v in PLAYER_VARIANTS:
            c = summary[f"{v}__vs__{opp_spec}"]
            totals[v][0] += c["wins"]
            totals[v][1] += c["n"]

    baseline_variants = [v for v in PLAYER_VARIANTS if v != "v046_final"]
    for baseline in baseline_variants:
        print(f"\n=== v046_final vs {baseline}: per-opponent delta ===")
        tw_new, tn_new = 0, 0
        tw_old, tn_old = 0, 0
        for opp_spec in OPPONENTS:
            c_new = summary[f"v046_final__vs__{opp_spec}"]
            c_old = summary[f"{baseline}__vs__{opp_spec}"]
            diff, pval = ztest(c_new["wins"], c_new["n"], c_old["wins"], c_old["n"])
            flag = ""
            if not math.isnan(pval):
                if pval < 0.05 and diff < 0:
                    flag = "  <-- REGRESSION"
                elif pval < 0.05 and diff > 0:
                    flag = "  <-- improvement"
            print(f"{opp_spec:26s} {baseline}={c_old['wr']:.3f}  v046_final={c_new['wr']:.3f}  "
                  f"delta={diff:+.3f}  p={pval:.4f}{flag}")
            tw_new += c_new["wins"]; tn_new += c_new["n"]
            tw_old += c_old["wins"]; tn_old += c_old["n"]
        diff, pval = ztest(tw_new, tn_new, tw_old, tn_old)
        wr_new = tw_new / tn_new if tn_new else float("nan")
        wr_old = tw_old / tn_old if tn_old else float("nan")
        print(f"POOLED: {baseline}={wr_old:.3f} ({tn_old} games)  v046_final={wr_new:.3f} ({tn_new} games)  "
              f"delta={diff:+.3f}  p={pval:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
