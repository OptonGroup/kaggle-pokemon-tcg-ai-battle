"""rng_contamination_probe.py — placebo test for in-process search RNG contamination.

Community finding (Kaggle notebook lucifer19/battlecore-compact-agent, recon 2026-07-18):
in a SHARED OS process, an agent-side search_begin/search_step call can perturb the live
engine's RNG stream even when the search's result is discarded (placebo arm measured
0.450 instead of the expected ~0.48 baseline). Our own gate infra (scripts/h3_gate.py,
src/eval/anchored_pool.py) runs games through a ProcessPoolExecutor pool WITHOUT
max_tasks_per_child, so a handful of long-lived worker processes each play MANY games
sequentially -- some with PIMCAgentV2's internal search_begin/search_step calls (via
cg_native.sim's single global `lib` handle), some without. If the underlying native
library shares RNG state across search and battle instances, our win-rate measurements
could carry cross-game contamination.

This probe runs the SAME matchup (v009a_crustle vs a generic greedy opponent, both with
ZERO decision-quality dependence on search) through two arms in the SAME process pool:
  ARM baseline: v009a plays normally, no search calls anywhere.
  ARM placebo:  v009a plays IDENTICALLY, but every decision first performs a genuine
                search_begin/search_step/search_end cycle (sampled hidden info, one
                arbitrary step) and DISCARDS the result before returning v009a's real
                choice unmodified.

If WR(baseline) and WR(placebo) diverge beyond what Wilson 95% CIs on this N would
explain by chance, that's evidence search calls perturb shared engine state even when
their results are thrown away -- direct analogue of the battlecore placebo finding.

Usage:
    .venv/bin/python scripts/rng_contamination_probe.py --games 200 --workers 4
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


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _load_submission_module(sub_dir_name: str):
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_rng_probe_{sub_dir_name}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


# ---------------------------------------------------------------------------
# Generic greedy opponent (dict-contract, mirrors scripts/h3_gate.greedy_meta_agent)
# ---------------------------------------------------------------------------

_ATTACK_DMG_MAP = None


def _dmg_map():
    global _ATTACK_DMG_MAP
    if _ATTACK_DMG_MAP is None:
        from cg_native.api import all_attack
        _ATTACK_DMG_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DMG_MAP


_OT_YES, _OT_ATTACK, _OT_END = 1, 13, 14
_ST_YES_NO, _ST_COUNT, _SC_MAIN = 9, 8, 0


def greedy_agent(obs_dict: dict) -> list[int]:
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


# ---------------------------------------------------------------------------
# Placebo search wrapper — genuine search_begin/search_step/search_end cycle,
# result discarded, real decision unaffected.
# ---------------------------------------------------------------------------

def _sample_hidden(known_ids: list[int], full_deck: list[int], count: int, rng: random.Random) -> list[int]:
    pool = list(full_deck)
    for kid in known_ids:
        try:
            pool.remove(kid)
        except ValueError:
            pass
    if len(pool) < count:
        pool.extend([4] * (count - len(pool)))
    if count == 0:
        return []
    return rng.sample(pool, min(count, len(pool)))


def _opp_known(obs, opp_idx: int) -> list[int]:
    known = []
    opp = obs.current.players[opp_idx]
    for c in (opp.discard or []):
        known.append(c.id)
    for p in list(opp.active or []) + list(opp.bench or []):
        if p is not None:
            known.append(p.id)
            for ec in (p.energyCards or []):
                known.append(ec.id)
    for c in (opp.prize or []):
        if c is not None:
            known.append(c.id)
    return known


def _make_placebo_wrapper(real_agent_fn, player_deck: list[int], opp_deck_full: list[int], rng: random.Random,
                           counters: dict):
    """Wraps real_agent_fn: every call first runs ONE genuine search_begin/search_step/
    search_end cycle (sampled hidden info, arbitrary step choice) and discards the
    result, then returns real_agent_fn(obs_dict) unmodified. counters tracks how many
    dummy searches actually executed vs. were skipped (malformed/error), so a null
    result (no measurable WR shift) can be distinguished from "the dummy search never
    actually ran"."""
    from cg_native.api import to_observation_class, search_begin, search_step, search_end

    def _wrapped(obs_dict: dict) -> list[int]:
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is not None:
                your_index = obs.current.yourIndex
                opp_idx = 1 - your_index
                my = obs.current.players[your_index]
                opp = obs.current.players[opp_idx]
                ydc = my.deckCount
                your_deck = (list(player_deck) + [4] * 10)[:max(ydc, 0)]
                your_prize = [p.id if p is not None else 4 for p in (my.prize or [])]
                opp_kn = _opp_known(obs, opp_idx)
                opp_deck_sampled = _sample_hidden(opp_kn, opp_deck_full, opp.deckCount, rng)
                opp_prize = []
                for p in (opp.prize or []):
                    if p is not None:
                        opp_prize.append(p.id)
                    else:
                        pool = [x for x in opp_deck_full if x not in opp_prize]
                        opp_prize.append(rng.choice(pool) if pool else 4)
                opp_hand = _sample_hidden(opp_kn, opp_deck_full, opp.handCount, rng)
                opp_active_ids = []
                if opp.active and len(opp.active) > 0 and opp.active[0] is None:
                    opp_active_ids = [rng.choice(opp_deck_full)]

                sst = search_begin(obs, your_deck, your_prize, opp_deck_sampled, opp_prize,
                                    opp_hand, opp_active_ids)
                n_opts = len(obs.select.option)
                if n_opts > 0:
                    search_step(sst.searchId, [0])
                search_end()
                counters["ok"] = counters.get("ok", 0) + 1
        except Exception:
            counters["fail"] = counters.get("fail", 0) + 1
        return real_agent_fn(obs_dict)

    return _wrapped


# ---------------------------------------------------------------------------
# Worker: plays one game, arm in {"baseline", "placebo"}
# ---------------------------------------------------------------------------

def _play_one_game(task: dict) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class
    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)

    from exit.pimc_agent import CRUSTLE_DECK_APPROX

    arm = task["arm"]
    our_first = task["our_first"]
    seed = task["seed"]
    rng = random.Random(seed)

    v009a_mod = _load_submission_module("v009a_crustle")
    player_deck = list(getattr(v009a_mod, "my_deck", None) or CRUSTLE_DECK_APPROX)
    opp_deck = list(CRUSTLE_DECK_APPROX)

    counters: dict = {}
    if arm == "placebo":
        player_agent = _make_placebo_wrapper(v009a_mod.agent, player_deck, opp_deck, rng, counters)
    else:
        player_agent = v009a_mod.agent

    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_agent, greedy_agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = greedy_agent, player_agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(arm=arm, seed=seed, our_first=our_first, outcome="crash", moves=0,
                    search_ok=counters.get("ok", 0), search_fail=counters.get("fail", 0))

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

    return dict(arm=arm, seed=seed, our_first=our_first, outcome=outcome, moves=move_count,
                search_ok=counters.get("ok", 0), search_fail=counters.get("fail", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="games per arm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260719)
    args = ap.parse_args()

    ctx = multiprocessing.get_context("spawn")
    queue = []
    for arm in ("baseline", "placebo"):
        for i in range(args.games):
            queue.append(dict(arm=arm, our_first=(i % 2 == 0), seed=args.seed + hash((arm, i)) % 1_000_000))
    random.Random(args.seed).shuffle(queue)  # interleave arms across the SAME worker pool

    print(f"[probe] {len(queue)} games total ({args.games}/arm), workers={args.workers}", flush=True)
    t0 = time.time()
    results = {"baseline": [], "placebo": []}
    search_ok_total = 0
    search_fail_total = 0

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_play_one_game, t): t for t in queue}
        n_done = 0
        for fut in futures:
            pass
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                t = futures.pop(f)
                try:
                    res = f.result()
                except Exception as e:
                    res = dict(arm=t["arm"], seed=t["seed"], our_first=t["our_first"],
                               outcome="crash", moves=0, search_ok=0, search_fail=0)
                results[res["arm"]].append(res)
                search_ok_total += res.get("search_ok", 0)
                search_fail_total += res.get("search_fail", 0)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  ... {n_done}/{len(queue)} done ({time.time()-t0:.0f}s)", flush=True)

    def summarize(rows):
        wins = sum(1 for r in rows if r["outcome"] == "our_win")
        losses = sum(1 for r in rows if r["outcome"] == "opp_win")
        draws = sum(1 for r in rows if r["outcome"] == "draw")
        aborted = sum(1 for r in rows if r["outcome"] in ("aborted", "crash"))
        n = wins + losses
        wr = wins / n if n else float("nan")
        lo, hi = wilson_ci(wins, n)
        return wins, losses, draws, aborted, n, wr, lo, hi

    print(f"\n=== RESULTS ({time.time()-t0:.0f}s total) ===")
    print(f"placebo dummy searches: ok={search_ok_total} fail={search_fail_total}")
    summary = {}
    for arm in ("baseline", "placebo"):
        wins, losses, draws, aborted, n, wr, lo, hi = summarize(results[arm])
        print(f"{arm:9s}: W={wins} L={losses} D={draws} aborted={aborted}  "
              f"WR={wr:.3f}  Wilson95=[{lo:.3f},{hi:.3f}]  n_decisive={n}")
        summary[arm] = dict(wins=wins, losses=losses, draws=draws, aborted=aborted, n=n, wr=wr, lo=lo, hi=hi)

    b, p = summary["baseline"], summary["placebo"]
    overlap = not (p["hi"] < b["lo"] or p["lo"] > b["hi"])
    delta = p["wr"] - b["wr"] if (not math.isnan(p["wr"]) and not math.isnan(b["wr"])) else float("nan")
    print(f"\ndelta (placebo - baseline) = {delta:+.3f}")
    print(f"Wilson CIs overlap: {overlap} {'(no evidence of contamination at this N)' if overlap else '(SUSPICIOUS — investigate further)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
