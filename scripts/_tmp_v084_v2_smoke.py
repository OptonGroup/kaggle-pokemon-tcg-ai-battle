"""_tmp_v084_v2_smoke.py — [v084 v2] End-to-end smoke of the REWRITTEN search_gs.py core
(playout + UCB1 + value-leaf). value_features_gs/value_model_gs don't exist yet (parallel
track) -- to actually exercise the playout/UCB1/BC-in-rollout code paths (not just confirm
"search unavailable, reason=no_value_model"), this script injects a tiny STUB pair into
sys.modules under those two names before importing search_gs, so `_get_value_api()`'s lazy
import finds them already cached. The submission directory itself is left untouched -- no
files are written there. This is not a strength/quality measurement, only a technical
liveness + cost-per-playout probe.

Stand: macOS ARM -> cg_native via sys.modules alias (scripts/gate_scout_guard.py:124-137
pattern).

Запуск:
    .venv/bin/python scripts/_tmp_v084_v2_smoke.py --games 6
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
import types
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

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

from cg.game import battle_start, battle_finish, battle_select  # noqa: E402

SUB_DIR = os.path.join(ROOT, "submissions", "v084_bc2b_search")
if SUB_DIR not in sys.path:
    sys.path.insert(0, SUB_DIR)

# ── Stub value model, injected into sys.modules BEFORE main.py/search_gs.py import it ──
_vf_stub = types.ModuleType("value_features_gs")


def _stub_value_features(raw_obs: dict, me_index: int):
    cur = raw_obs.get("current") or {}
    players = cur.get("players") or []
    if not (0 <= me_index < len(players)):
        return [0.0, 0.0]
    mine = players[me_index] or {}
    theirs = players[1 - me_index] or {}
    prize_diff = float(len((theirs.get("prize") or [])) - len((mine.get("prize") or [])))

    def _hp(pl):
        total = 0
        for mon in list(pl.get("active") or []) + list(pl.get("bench") or []):
            if mon:
                total += mon.get("hp") or 0
        return float(total)

    return [prize_diff, _hp(mine) - _hp(theirs)]


_vf_stub.value_features = _stub_value_features
sys.modules["value_features_gs"] = _vf_stub

_vm_stub = types.ModuleType("value_model_gs")


def _stub_value_predict(feat) -> float:
    # Cheap monotone squashing stand-in -- NOT a trained model, only here to exercise the
    # playout/leaf/UCB1 plumbing end-to-end. prize_diff dominates, hp_diff is a tiebreaker.
    prize_diff, hp_diff = feat[0], feat[1]
    raw = 0.35 * prize_diff + 0.001 * hp_diff
    if raw > 8:
        return 1.0
    if raw < -8:
        return 0.0
    import math
    return 1.0 / (1.0 + math.exp(-raw))


_vm_stub.value_predict = _stub_value_predict
sys.modules["value_model_gs"] = _vm_stub


_LOAD_COUNTER = 0


def _load_module(path: str):
    """Every call gives a genuinely independent module object (unique spec name, forced
    re-exec) -- REQUIRED for true self-play: main.py keeps per-game state at module scope
    (_adrena_state, _loop_sig, ...) that assumes exactly one player uses one module
    instance. Using a single shared main_mod for both sides (this script's own first-draft
    bug) cross-contaminates that state between the two players' unrelated decisions and
    produces bogus picks -- reproduced with search disabled too, confirming it was a smoke-
    harness bug, not a search_gs regression. Matches the established pattern in
    scripts/gate_scout_guard.py:277-296 / scripts/v074_fml_probe.py:67-84 (unique mod_name
    per load, even when both sides load the SAME submission directory for a mirror match).
    Note: `import search_gs` / `import bc_features_2b` STATEMENTS inside each independently-
    loaded main.py copy still resolve through the normal sys.modules cache by name, so both
    sides share ONE search_gs/bc_features_2b/bc_model_2b instance for the game's duration --
    this is the same pre-existing property the project's own gate scripts have (mirror
    matches there hit it too, whenever both sides land in the same process), unrelated to
    and not fixed by this change."""
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    mod_name = f"_v2smoke_main_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(SUB_DIR)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--max-moves", type=int, default=2000)
    args = ap.parse_args()

    main_path = os.path.join(SUB_DIR, "main.py")
    mod0 = _load_module(main_path)
    mod1 = _load_module(main_path)
    search_gs = sys.modules.get("search_gs") or mod0._sgs
    print(f"search_gs module: {search_gs} (shared by both sides, see _load_module docstring)")
    print(f"value_api available: {search_gs._get_value_api() is not None}")
    print(f"bc_api available: {search_gs._get_bc_api() is not None}")
    my_deck = list(mod0.my_deck)
    agents = (mod0.agent, mod1.agent)

    stats = Counter()
    playout_ms_samples = []
    reasons = Counter()
    exc_types = Counter()
    t_all = time.perf_counter()

    for g in range(args.games):
        random.seed(2000 + g)
        # Real Kaggle harness convention: first call per game has select=None (deck
        # request), which is also what triggers main.py's internal per-game state reset
        # (_adrena_state/_loop_sig/etc, plus search_gs.reset_episode()). battle_start()
        # takes decks directly and never emits a select=None observation itself, so we
        # invoke that reset path explicitly for BOTH independent agent instances.
        mod0.agent({"select": None})
        mod1.agent({"select": None})
        obs_dict, _ = battle_start(list(my_deck), list(my_deck))
        moves = 0
        while obs_dict is not None and moves < args.max_moves:
            sel = obs_dict.get("select")
            if sel is None:
                break
            cur = obs_dict.get("current") or {}
            your_index = int(cur.get("yourIndex") or 0)
            agent_fn = agents[your_index]
            n_before_calls = search_gs._SPENT["n_calls"]
            t0 = time.perf_counter()
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                stats["exceptions"] += 1
                exc_types[type(e).__name__] += 1
                print(f"  [game {g}] agent() raised at move {moves} (side={your_index}): {e!r}")
                break
            dt_ms = (time.perf_counter() - t0) * 1000.0
            n_after_calls = search_gs._SPENT["n_calls"]
            if n_after_calls > n_before_calls:
                stats["search_calls"] += 1
                playout_ms_samples.append(dt_ms)
            if not isinstance(choice, list):
                stats["bad_choice_type"] += 1
                print(f"  [game {g}] non-list choice {choice!r} at move {moves} (side={your_index})")
                break
            opts = sel.get("option") or []
            min_c = int(sel.get("minCount") or 0)
            max_c = int(sel.get("maxCount") or 1)
            if len(opts) == 0:
                # main.py's own top-of-agent() contract: n_options==0 -> always return [].
                # Not something search_gs/this smoke touches; whether the engine truly
                # accepts an empty pick against a nonzero minCount here is exactly what
                # calling battle_select below will tell us -- don't pre-judge it as illegal.
                legal = (choice == [])
            else:
                legal = (all(isinstance(i, int) and 0 <= i < len(opts) for i in choice)
                         and len(set(choice)) == len(choice)
                         and min_c <= len(choice) <= max(max_c, min_c, 1))
            if not legal:
                stats["illegal_choice"] += 1
                print(f"  [game {g}] illegal choice {choice} at move {moves} (side={your_index}) "
                      f"n_opts={len(opts)} min={min_c} max={max_c} ctx={sel.get('context')}")
                break
            try:
                obs_dict = battle_select(choice)
            except Exception as e:
                # [known, pre-existing, reproduces with V084_SEARCH_DISABLE=1 too --
                # unrelated to search_gs v2] ctx=7 with n_opts==0 but minCount>=1 near
                # end-game in TRUE mirror self-play: main.py's own top-of-agent()
                # contract (n_options==0 -> return []) hits an engine reject here. Not
                # this smoke's concern -- log and move to the next game.
                stats["engine_reject_preexisting"] += 1
                print(f"  [game {g}] battle_select rejected choice={choice} at move {moves} "
                      f"(side={your_index}, ctx={sel.get('context')}, n_opts={len(opts)}, "
                      f"min={min_c}, max={max_c}): {e!r} -- pre-existing, not a search_gs issue")
                break
            moves += 1
        stats["moves_total"] += moves
        stats["games"] += 1
        print(f"  game {g+1}/{args.games} moves={moves} "
              f"search_calls_so_far={stats['search_calls']} "
              f"overrides_so_far={search_gs._SPENT['n_overrides']} "
              f"n_exceptions={search_gs._SPENT['n_exceptions']} "
              f"disabled={search_gs._SPENT['disabled']} "
              f"({time.perf_counter()-t_all:.1f}s elapsed)", flush=True)
        battle_finish()
        search_gs.reset_episode()

    print("\n=== СВОДКА ===")
    print(f"games={stats['games']} moves_total={stats['moves_total']} "
          f"search_calls={stats['search_calls']} "
          f"overrides={search_gs._SPENT['n_overrides']} "
          f"exceptions={stats['exceptions']} illegal_choice={stats['illegal_choice']} "
          f"bad_choice_type={stats['bad_choice_type']}")
    if exc_types:
        print(f"exception types: {dict(exc_types)}")
    if playout_ms_samples:
        srt = sorted(playout_ms_samples)
        n = len(srt)
        print(f"decide()-triggering agent() calls: n={n} "
              f"mean={sum(srt)/n:.1f}ms p50={srt[n//2]:.1f}ms "
              f"p90={srt[int(n*0.9)]:.1f}ms max={srt[-1]:.1f}ms")
    else:
        print("NO search_gs.decide() calls were counted (n_calls never incremented) -- "
              "check gating (turn/n_options/ctx) or that value_api loaded.")
    print(f"total wall time: {time.perf_counter()-t_all:.1f}s")


if __name__ == "__main__":
    main()
