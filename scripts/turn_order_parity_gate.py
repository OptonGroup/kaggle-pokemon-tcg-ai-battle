#!/usr/bin/env python3
"""turn_order_parity_gate.py — parity A/B measuring the effect of the IS_FIRST
(«хотите ходить первым?») default choice, isolated to that ONE bit.

Motivation: `submissions/v076_final/scoring_grimmsnarl.py:855-856` hard-codes
"choose second" (same pattern in v036/v046/v048/v050 — checked). Форум 723591
(5333 игр) says going first is worth ~+10пп WR pooled (55.2% vs 44.8%), with the
gap deck-specific (brief: Alakazam +8.9пп, Dragapult +1.5пп). Before touching any
submission this script measures the isolated effect via a parity gate: TWO copies
of the exact same v076_final agent code, differing ONLY in what they answer when
the engine presents the IS_FIRST select — one forced to always answer "first",
the other forced to always answer "second" (matching the current hard-code) — each
played N games against the SAME fixed, deterministic, local opponent
(`greedy_meta_agent`, ported from scripts/gate_scout_guard.py — always answers YES
on any YES/NO select, so ITS own IS_FIRST behaviour is identical across both arms
and cannot confound the comparison). Run separately per opponent deck (per-deck
effect, not pooled — Simpson's-paradox trap called out in
scripts/scout_gate_paired.py's docstring applies here too).

Mechanics (verified against submissions/v076_final and src/cg_native/api.py):
SelectContext.IS_FIRST(=41) is a SelectType.YES_NO(=9) select shown ONLY to
whichever side wins the engine's internal coin flip (unseeded, not controllable
from Python — see gate_scout_guard.py's note on battle_start not taking a seed).
Consequently only ~50% of games in each arm actually present the candidate with
this choice; the other ~50% the FIXED opponent won the toss and (always chooses
"first" itself) takes the first move regardless of our arm. The unconditional WR
delta measured here is therefore a lower bound / diluted estimate of the
conditional ("given you have the choice") effect — expect roughly half the pooled
forum delta at best. Reported honestly, not adjusted.

Engine: cg_native (macOS arm64 local build), multiprocessing with the "spawn"
context (per project convention — threading is confirmed to SIGSEGV this engine,
never use it here).

Usage:
    .venv/bin/python scripts/turn_order_parity_gate.py --smoke            # N=20 sanity
    .venv/bin/python scripts/turn_order_parity_gate.py --n 300 --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
    from cg_native.api import to_observation_class  # noqa: E402
    import cg_native  # noqa: E402
    import cg_native.api, cg_native.game, cg_native.sim, cg_native.utils  # noqa: E402
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)
    ENGINE = "cg_native"
except Exception:
    from cg.game import battle_start, battle_finish, battle_select  # noqa: E402
    from cg.api import to_observation_class  # noqa: E402
    ENGINE = "cg"

MAX_MOVES_PER_GAME = 2000
GATES_DIR = os.path.join(ROOT, "results", "gates")
CANDIDATE_DIR = os.path.join(ROOT, "submissions", "v076_final")

# SelectContext.IS_FIRST / SelectType.YES_NO / OptionType.{YES,NO} — src/cg_native/api.py.
_SC_IS_FIRST = 41
_ST_YES_NO = 9
_OT_YES = 1
_OT_NO = 2

_OT_ATTACK = 13
_GREEDY_PRIORITY = (13, 10, 9, 7, 8, 12, 3, 14)  # ATTACK,ABILITY,EVOLVE,PLAY,ATTACH,RETREAT,CARD,END

_ATTACK_DAMAGE_MAP = None


def _opt_get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _get_attack_damage_map():
    global _ATTACK_DAMAGE_MAP
    if _ATTACK_DAMAGE_MAP is None:
        try:
            from cg_native.api import all_attack
        except ImportError:
            from cg.api import all_attack  # type: ignore[no-redef]
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def greedy_meta_agent(obs_dict: dict) -> list[int]:
    """Fixed, deterministic, deck-agnostic local opponent (ported verbatim from
    scripts/gate_scout_guard.py so this script has no import-order dependency on
    it). Always answers YES on any YES/NO select — including IS_FIRST — so its
    own turn-order behaviour is IDENTICAL across both candidate arms below and
    cannot confound the A/B."""
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
            if _opt_get(opts[i], "type") == _OT_YES:
                return [i]
        return [0]
    if sel_type == 8:  # SelectType.COUNT
        best = max(range(n), key=lambda i: (_opt_get(opts[i], "number", 0) or 0))
        return [best]
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    if sel.get("context") == 0:  # SelectContext.MAIN
        atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
            return [best]
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min(min_c, max_c, n))
    return list(range(k))


def _load_candidate_agent():
    """Fresh v076_final module per game (fresh globals — same hygiene as
    gate_scout_guard.py's _load_submission_module: v076_final keeps mutable
    per-game state in module globals, e.g. _adrena_state)."""
    import importlib.util
    main_path = os.path.join(CANDIDATE_DIR, "main.py")
    if CANDIDATE_DIR not in sys.path:
        sys.path.insert(0, CANDIDATE_DIR)
    global _LOAD_COUNTER
    _LOAD_COUNTER = globals().get("_LOAD_COUNTER", 0) + 1
    mod_name = f"_turnorder_v076_{_LOAD_COUNTER}"
    cwd = os.getcwd()
    try:
        os.chdir(CANDIDATE_DIR)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    # v076_final logs one "[dec]"/"[alt]" line per decision to stderr (DECISION_LOG=True,
    # scoring_grimmsnarl.py ~line 1754 — for post-hoc Kaggle log analysis, irrelevant here
    # and expensive across hundreds of games). Silenced at runtime via the module attribute,
    # NOT by editing submissions/v076_final on disk.
    scorer = getattr(mod, "_scorer", None)
    if scorer is not None:
        try:
            scorer.DECISION_LOG = False
        except Exception:
            pass
    return mod


def _make_forced_turn_order_agent(real_agent, mode: str):
    """Wraps a real agent(obs_dict) fn: on the IS_FIRST select, force the answer
    to `mode` ("first"/"second"); every other decision is delegated UNCHANGED to
    the real agent. This is the parity property: the two arms are byte-identical
    code except for this one intercepted select."""
    want_type = _OT_YES if mode == "first" else _OT_NO

    def wrapped(obs_dict):
        sel = obs_dict.get("select") if obs_dict else None
        if sel is not None and sel.get("context") == _SC_IS_FIRST and sel.get("type") == _ST_YES_NO:
            opts = sel.get("option") or []
            for i, o in enumerate(opts):
                if _opt_get(o, "type") == want_type:
                    return [i]
            return [0] if opts else []
        return real_agent(obs_dict)

    return wrapped


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(ids)}")
    return ids


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_prop_ztest(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return (0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1 - p_pool), 1e-12) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, math.erfc(abs(z) / math.sqrt(2)))


def _is_legal(choice, sel) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = 0 if sel.get("minCount") is None else int(sel.get("minCount"))
    max_c = 1 if sel.get("maxCount") is None else int(sel.get("maxCount"))
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


def _play_one_game(task: dict) -> dict:
    mode = task["mode"]
    opp_deck_name = task["opp_deck_name"]
    cand_first_seat = task["cand_first_seat"]
    seed = task["seed"]

    import random
    random.seed(seed)

    cand_mod = _load_candidate_agent()
    cand_deck = list(cand_mod.my_deck)
    cand_agent = _make_forced_turn_order_agent(cand_mod.agent, mode)

    opp_deck = _load_deck_csv(os.path.join(ROOT, "decks", "meta", f"{opp_deck_name}.csv"))
    opp_agent = greedy_meta_agent

    if cand_first_seat:
        deck0, deck1 = cand_deck, list(opp_deck)
        agent0, agent1 = cand_agent, opp_agent
    else:
        deck0, deck1 = list(opp_deck), cand_deck
        agent0, agent1 = opp_agent, cand_agent

    t0 = time.perf_counter()
    obs_dict, _ = battle_start(deck0, deck1)
    if obs_dict is None:
        return dict(mode=mode, opp_deck_name=opp_deck_name, outcome="crash",
                    exc_side=None, illegal_side=None, moves=0, elapsed_s=time.perf_counter() - t0)

    move_count = 0
    n_illegal = 0
    illegal_side = None
    n_exceptions = 0
    exc_side = None
    winner = None
    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            is_cand_turn = (your_index == 0) == cand_first_seat
            agent_fn = agent0 if your_index == 0 else agent1
            sel = obs_dict.get("select")
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                exc_side = "cand" if is_cand_turn else "opp"
                break
            if not _is_legal(choice, sel):
                n_illegal += 1
                illegal_side = "cand" if is_cand_turn else "opp"
                break
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    cand_side = 0 if cand_first_seat else 1
    aborted = winner is None and n_exceptions == 0 and n_illegal == 0
    if n_exceptions:
        outcome = "exception"
    elif n_illegal:
        outcome = "illegal"
    elif aborted:
        outcome = "timeout"
    elif winner == cand_side:
        outcome = "cand_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return dict(mode=mode, opp_deck_name=opp_deck_name, outcome=outcome, exc_side=exc_side,
                illegal_side=illegal_side, moves=move_count, elapsed_s=time.perf_counter() - t0)


def run_arm(mode: str, opp_deck_name: str, n_games: int, base_seed: int, workers: int) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue = []
    for game_idx in range(n_games):
        cand_first_seat = (game_idx % 2 == 0)
        seed = base_seed + (zlib.crc32(f"{mode}|{opp_deck_name}|{game_idx}".encode()) % 1_000_000)
        queue.append(dict(mode=mode, opp_deck_name=opp_deck_name, cand_first_seat=cand_first_seat,
                           seed=seed, game_idx=game_idx))

    t_start = time.time()
    games = []
    q_iter = iter(queue)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {}
        for _ in range(workers):
            try:
                t = next(q_iter)
            except StopIteration:
                break
            futures[ex.submit(_play_one_game, t)] = t
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                t = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(mode=mode, opp_deck_name=opp_deck_name, outcome="worker_crash",
                               exc_side="cand", illegal_side=None, moves=0, elapsed_s=0.0, error=str(e))
                games.append(res)
                try:
                    t2 = next(q_iter)
                    futures[ex.submit(_play_one_game, t2)] = t2
                except StopIteration:
                    pass
    elapsed = time.time() - t_start

    wins = sum(1 for g in games if g["outcome"] == "cand_win")
    losses = sum(1 for g in games if g["outcome"] == "opp_win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    exc = sum(1 for g in games if g["outcome"] in ("exception", "worker_crash", "crash"))
    illegal = sum(1 for g in games if g["outcome"] == "illegal")
    timeouts = sum(1 for g in games if g["outcome"] == "timeout")

    n = len(games)
    we = wins + 0.5 * draws
    win_rate = we / n if n else 0.0
    wlo, whi = wilson_ci(we, n)

    return dict(mode=mode, opp_deck_name=opp_deck_name, n_games=n, wins=wins, losses=losses,
                draws=draws, win_equivalent=we, win_rate=win_rate, wilson95_lo=wlo, wilson95_hi=whi,
                exceptions=exc, illegal=illegal, timeouts=timeouts, elapsed_s=elapsed,
                games_per_sec=(n / elapsed if elapsed > 0 else 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--decks", nargs="+", default=["alakazam_aidy", "Dragapult"])
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--smoke", action="store_true", help="override --n to 20")
    ap.add_argument("--out", default=os.path.join(GATES_DIR, "turn_order_parity.json"))
    args = ap.parse_args()
    n_games = 20 if args.smoke else args.n

    print(f"engine={ENGINE} candidate={CANDIDATE_DIR} n_per_arm={n_games} workers={args.workers}")
    results = {}
    for deck_name in args.decks:
        arm_results = {}
        for mode in ("first", "second"):
            print(f"--- deck={deck_name} mode={mode} ---")
            r = run_arm(mode, deck_name, n_games, args.seed, args.workers)
            arm_results[mode] = r
            print(f"  n={r['n_games']} wins={r['wins']} losses={r['losses']} draws={r['draws']} "
                  f"WR={r['win_rate']:.4f} CI95=[{r['wilson95_lo']:.4f},{r['wilson95_hi']:.4f}] "
                  f"exc={r['exceptions']} illegal={r['illegal']} timeouts={r['timeouts']} "
                  f"{r['games_per_sec']:.1f} games/s")
        wf, wsec = arm_results["first"], arm_results["second"]
        z, p = two_prop_ztest(wf["win_equivalent"], wf["n_games"], wsec["win_equivalent"], wsec["n_games"])
        delta = wf["win_rate"] - wsec["win_rate"]
        print(f"  DELTA(first-second) = {delta:+.4f}  z={z:.3f} p={p:.4f}")
        results[deck_name] = dict(first=wf, second=wsec, delta_wr=delta, z=z, p_value=p)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
