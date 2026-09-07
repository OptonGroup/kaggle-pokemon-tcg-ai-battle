"""anchored_pool.py — Gauntlet-v3: anchored evaluation pool + predicted-mu calibration.

PLAN.md §H2. Purpose: local gates vs our own agents do NOT predict the live
ladder (v012: local 0.70 -> live 499; v018: 66% vs Alakazam -> live 484).
This script builds a reusable round-robin among 7 anchor submissions with
KNOWN live-mu (frozen 2026-07-11) and calibrates a Bradley-Terry-style scalar
`s` mapping local win-rate deltas to mu deltas:

    WR(i, j) ~= 1 / (1 + 10 ** (-(mu_i - mu_j) / s))

so a future candidate's win-rates vs the 7 anchors can be converted into a
predicted-mu via the same MLE used in the leave-one-out (LOO) validation here.

Engine: ONLY the native macOS arm64 wrapper `src/cg_native` (per guardrails).
Threading is UNSAFE (data corruption) -> parallelism is multiprocessing
(spawn) only, workers default 4. The engine's PRNG is not seedable in a
useful way (results are non-deterministic even with a fixed seed passed to
Python's `random` module) -- `--seed` only seeds side-alternation bookkeeping
and Python-side `random`, not engine outcomes.

Resumable: every finished game is appended to a state JSON
(`results/gates/anchored_pool_state.json` by default) immediately, so a
Bash call that hits its ~10-minute wall can be killed/re-run freely -- the
next invocation picks up exactly where it left off (round-robin-interleaved
task queue, so a partial run still has *some* data on every pair rather than
completing pairs strictly in order).

Usage:
    # smoke: 1 pair, 10 games, 2 workers
    .venv/bin/python -m src.eval.anchored_pool --roundrobin \
        --anchors v007_crustle,v012_iono_heur --games-per-pair 10 \
        --workers 2 --max-minutes 5

    # full round robin, chunked (run repeatedly until "COMPLETE")
    .venv/bin/python -m src.eval.anchored_pool --roundrobin \
        --games-per-pair 60 --workers 4 --max-minutes 8

    # candidate vs all 7 anchors (future H3+ use)
    .venv/bin/python -m src.eval.anchored_pool --candidate submissions/v0XX_foo \
        --games-per-pair 100 --workers 4 --max-minutes 8

    # calibration (reads state, no games played)
    .venv/bin/python -m src.eval.anchored_pool --calibrate
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Native engine ONLY (macOS arm64). Never import src/cg (Linux-only .so).
from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
import cg_native  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

# ---------------------------------------------------------------------------
# CRITICAL macOS shim: every submissions/vNNN_*/cg/ dir bundles the Kaggle
# runtime's LINUX x86_64 libcg.so (for Kaggle upload). Some agents (v001,
# v002, v012) do `from cg.api import ...`, which would try to dlopen that
# Linux .so and crash on macOS ARM. Others (v007/v009a/v015/v017/v018) avoid
# importing `cg` entirely and were unaffected -- but a general-purpose pool
# script must not depend on which agents happen to sidestep this. Fix: alias
# the `cg` package name in sys.modules to the already-imported NATIVE
# cg_native package/submodules *before* any agent main.py is exec'd. Python's
# import machinery checks sys.modules first for each dotted component, so
# `from cg.api import X` inside a submission's main.py resolves to the
# already-loaded native module object and never touches the bundled Linux
# .so or re-executes any file from submissions/*/cg/.
# ---------------------------------------------------------------------------
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

MAX_MOVES_PER_GAME = 2000

GATES_DIR = os.path.join(ROOT, "results", "gates")
STATE_PATH_DEFAULT = os.path.join(GATES_DIR, "anchored_pool_state.json")
CALIBRATION_PATH = os.path.join(GATES_DIR, "anchored_calibration.json")

# ---------------------------------------------------------------------------
# Anchors: live-mu frozen 2026-07-11 (PLAN.md §H2-1). v002_lucario flagged as
# a weak/stale anchor (June snapshot, meta has since rotated) -- kept in the
# "all-7" calibration variant but dropped in the "no-v002" variant.
# ---------------------------------------------------------------------------
ANCHOR_MU: dict[str, float] = {
    "v015_exit_pilot": 317.7,
    "v001_random": 483.8,
    "v018_grimmsnarl": 484.1,
    "v012_iono_heur": 499.2,
    "v009a_crustle": 724.1,
    "v007_crustle": 733.1,
    "v002_lucario": 779.9,
}
ANCHOR_NAMES: list[str] = list(ANCHOR_MU.keys())
WEAK_ANCHOR = "v002_lucario"

# Plan's expected coarse ordering (gross-inversion check, §H2-2 criterion)
CHAIN = ["v007_crustle", "v009a_crustle", "v012_iono_heur", "v018_grimmsnarl", "v015_exit_pilot"]


# ---------------------------------------------------------------------------
# Small stats helpers (no scipy; math/itertools only)
# ---------------------------------------------------------------------------

def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def smoothed_wr(we: float, n: float) -> float:
    """Empirical win fraction, Laplace-smoothed (+1/+2) only at the 0%/100% edges
    (per PLAN.md §H2-2) to avoid infinite log-loss; otherwise the raw MLE."""
    if n <= 0:
        return 0.5
    wr = we / n
    if wr <= 0.0 or wr >= 1.0:
        return (we + 1.0) / (n + 2.0)
    return wr


def _log_grid(lo: float, hi: float, num: int) -> list[float]:
    if num < 2:
        return [lo]
    ratio = (hi / lo) ** (1.0 / (num - 1))
    return [lo * (ratio ** i) for i in range(num)]


def _golden_min(f, a: float, b: float, tol: float = 1e-3, max_iter: int = 200) -> float:
    """1-D unimodal minimizer, no scipy."""
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        it += 1
    return (a + b) / 2


def spearman(x: list[float], y: list[float]) -> float:
    """Manual Spearman rank correlation (average ranks for ties)."""
    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for kk in range(i, j + 1):
                ranks[order[kk]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    n = len(x)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    varx = sum((r - mean_rx) ** 2 for r in rx)
    vary = sum((r - mean_ry) ** 2 for r in ry)
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / math.sqrt(varx * vary)


# ---------------------------------------------------------------------------
# Submission resolution / deck / agent loading
# ---------------------------------------------------------------------------

def resolve_sub_dir(spec: str) -> str:
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    cand = os.path.join(ROOT, spec)
    if os.path.isdir(cand):
        return cand
    cand2 = os.path.join(ROOT, "submissions", spec)
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(f"Не найдена директория сабмита: {spec!r}")


def load_deck_csv(sub_dir: str) -> list[int]:
    path = os.path.join(sub_dir, "deck.csv")
    with open(path) as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(ids)}")
    return ids


_LOAD_COUNTER = 0


def _load_agent(sub_dir: str, tag: str):
    """Fresh module instance every call (no cross-game module-global state
    leakage -- e.g. v018_grimmsnarl's `_adrena_state`). Same pattern as
    scripts/pimc_field_gauntlet.py / src/exit/opponents.py._load_agent_module."""
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    main_path = os.path.join(sub_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(main_path)
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_anchor_pool_{tag}_{os.getpid()}_{_LOAD_COUNTER}"
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)  # some agents read "deck.csv"/"EN_Card_Data.csv" relative to CWD
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path  # Kaggle-safe attr some agents rely on for path resolution
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No agent() in {main_path}")
    return mod


def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
    max_c = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


# ---------------------------------------------------------------------------
# One game — runs inside a worker process. Task/result are plain dict/str/
# int/bool/float only (pickle-safe for ProcessPoolExecutor + spawn).
# ---------------------------------------------------------------------------

def _play_one_game(task: dict) -> dict:
    left_dir = task["left_dir"]
    right_dir = task["right_dir"]
    left_first = task["left_first"]
    seed = task["seed"]

    import random
    random.seed(seed)

    try:
        left_mod = _load_agent(left_dir, "L")
        right_mod = _load_agent(right_dir, "R")
        left_deck = load_deck_csv(left_dir)
        right_deck = load_deck_csv(right_dir)
    except Exception as e:
        return dict(status="load_error", error=repr(e), outcome="aborted",
                     moves=0, wall_time=0.0, exception_who=None, **_task_ids(task))

    if left_first:
        deck0, deck1 = left_deck, right_deck
        agent0, agent1 = left_mod.agent, right_mod.agent
    else:
        deck0, deck1 = right_deck, left_deck
        agent0, agent1 = right_mod.agent, left_mod.agent

    t0 = time.perf_counter()
    winner_side = None  # 0/1/2(draw)/None(aborted)
    exc_side = None
    n_moves = 0

    try:
        obs_dict, start = battle_start(list(deck0), list(deck1))
        if obs_dict is None:
            return dict(status="engine_reject", error=f"errorType={getattr(start, 'errorType', None)}",
                         outcome="aborted", moves=0, wall_time=0.0, exception_who=None, **_task_ids(task))

        while n_moves < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner_side = obs.current.result
                break

            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            sel = obs_dict.get("select")

            try:
                choice = agent_fn(obs_dict)
            except Exception:
                exc_side = your_index
                winner_side = 1 - your_index
                break

            if not _is_legal(choice, sel):
                exc_side = your_index
                winner_side = 1 - your_index
                break

            try:
                obs_dict = battle_select(choice)
            except Exception:
                exc_side = your_index
                winner_side = 1 - your_index
                break

            n_moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    wall = time.perf_counter() - t0

    if winner_side is None:
        outcome = "aborted"  # hit MAX_MOVES_PER_GAME without a resolution
    elif winner_side == 2:
        outcome = "draw"
    else:
        engine_left_side = 0 if left_first else 1
        outcome = "left_win" if winner_side == engine_left_side else "right_win"

    exception_who = None
    if exc_side is not None:
        engine_left_side = 0 if left_first else 1
        exception_who = "left" if exc_side == engine_left_side else "right"

    return dict(status="ok", outcome=outcome, moves=n_moves, wall_time=wall,
                exception_who=exception_who, **_task_ids(task))


def _task_ids(task: dict) -> dict:
    return dict(pair_key=task["pair_key"], left_name=task["left_name"], right_name=task["right_name"],
                left_first=task["left_first"], game_idx=task["game_idx"], seed=task["seed"])


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"roundrobin": {}, "candidate": {}}


def save_state(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def pair_key(a: str, b: str) -> str:
    return f"{a}__vs__{b}"


def all_pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(names, 2))


def cell_summary(games: list[dict]) -> dict:
    left_wins = sum(1 for g in games if g.get("outcome") == "left_win")
    right_wins = sum(1 for g in games if g.get("outcome") == "right_win")
    draws = sum(1 for g in games if g.get("outcome") == "draw")
    aborted = sum(1 for g in games if g.get("outcome") == "aborted" and g.get("status") == "ok")
    infra_errors = sum(1 for g in games if g.get("status") not in (None, "ok"))
    exceptions_left = sum(1 for g in games if g.get("exception_who") == "left")
    exceptions_right = sum(1 for g in games if g.get("exception_who") == "right")

    n_decisive_draw = left_wins + right_wins + draws
    left_we = left_wins + 0.5 * draws
    wr_left = left_we / n_decisive_draw if n_decisive_draw else float("nan")
    lo, hi = wilson_ci(left_we, n_decisive_draw) if n_decisive_draw else (float("nan"), float("nan"))

    return dict(
        n=len(games), left_wins=left_wins, right_wins=right_wins, draws=draws,
        aborted=aborted, infra_errors=infra_errors,
        exceptions_left=exceptions_left, exceptions_right=exceptions_right,
        decisive_and_draw=n_decisive_draw, wr_left=wr_left,
        wilson_lo=lo, wilson_hi=hi,
    )


def record_result(state: dict, section: str, task: dict, res: dict) -> None:
    state.setdefault(section, {})
    pk = task["pair_key"]
    cell = state[section].setdefault(pk, {"left": task["left_name"], "right": task["right_name"], "games": []})
    entry = {
        "game_idx": task["game_idx"],
        "left_first": task["left_first"],
        "seed": task["seed"],
        "status": res.get("status"),
        "outcome": res.get("outcome"),
        "moves": res.get("moves"),
        "wall_time": res.get("wall_time"),
        "exception_who": res.get("exception_who"),
    }
    if res.get("status") != "ok":
        entry["error"] = res.get("error")
    cell["games"].append(entry)


# ---------------------------------------------------------------------------
# Task queue: round-robin-interleaved so a partial run still touches every
# pair rather than exhausting pair #1 fully before starting pair #2.
# ---------------------------------------------------------------------------

def build_task_queue(state: dict, section: str, pairs: list[tuple[str, str]],
                      games_target: int, base_seed: int) -> list[dict]:
    pending = []
    for a, b in pairs:
        k = pair_key(a, b)
        have = len(state.get(section, {}).get(k, {}).get("games", []))
        need = max(0, games_target - have)
        pending.append([a, b, k, have, need, 0])  # last field = local counter

    queue: list[dict] = []
    progressed = True
    while progressed:
        progressed = False
        for row in pending:
            a, b, k, have, need, ctr = row
            if ctr < need:
                game_idx = have + ctr
                left_first = (game_idx % 2 == 0)
                seed = base_seed + (hash((k, game_idx)) % 1_000_000)
                queue.append(dict(
                    pair_key=k, left_name=a, right_name=b,
                    left_dir=resolve_sub_dir(a), right_dir=resolve_sub_dir(b),
                    left_first=left_first, game_idx=game_idx, seed=seed,
                ))
                row[5] += 1
                progressed = True
    return queue


def run_queue(state: dict, section: str, queue: list[dict], state_path: str,
              workers: int, deadline: float, max_games_this_call: int | None) -> int:
    """Runs tasks from `queue` via a spawn ProcessPoolExecutor, saving state
    after every completed game. Stops issuing NEW tasks once the deadline (or
    max_games_this_call) is hit, but drains in-flight tasks before returning."""
    if not queue:
        return 0

    ctx = multiprocessing.get_context("spawn")
    q_iter = iter(queue)
    n_started = 0
    n_finished = 0

    def _can_submit_more() -> bool:
        if time.time() >= deadline:
            return False
        if max_games_this_call is not None and n_started >= max_games_this_call:
            return False
        return True

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {}
        for _ in range(workers):
            if not _can_submit_more():
                break
            try:
                t = next(q_iter)
            except StopIteration:
                break
            futures[ex.submit(_play_one_game, t)] = t
            n_started += 1

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                t = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(status="worker_crash", error=repr(e), outcome="aborted",
                                moves=0, wall_time=0.0, exception_who=None)
                record_result(state, section, t, res)
                save_state(state, state_path)
                n_finished += 1
                status = res.get("status")
                out = res.get("outcome")
                tag = "" if status == "ok" else f" [{status}: {res.get('error')}]"
                print(f"  [{t['pair_key']}] игра {t['game_idx']:3d} left_first={t['left_first']!s:5s} "
                      f"outcome={out:9s} moves={res.get('moves', 0):4d} "
                      f"wall={res.get('wall_time', 0.0):5.1f}s{tag}", flush=True)

                if _can_submit_more():
                    try:
                        nt = next(q_iter)
                        futures[ex.submit(_play_one_game, nt)] = nt
                        n_started += 1
                    except StopIteration:
                        pass
    return n_finished


# ---------------------------------------------------------------------------
# Matrix / report building
# ---------------------------------------------------------------------------

def build_matrix(state: dict, names: list[str]) -> tuple[dict, dict, int]:
    matrix: dict[str, dict[str, float | None]] = {a: {b: None for b in names} for a in names}
    pairs_summary: dict[str, dict] = {}
    exceptions_total = 0
    for a, b in all_pairs(names):
        k = pair_key(a, b)
        games = state.get("roundrobin", {}).get(k, {}).get("games", [])
        summ = cell_summary(games)
        pairs_summary[k] = summ
        wr = summ["wr_left"]
        matrix[a][b] = None if math.isnan(wr) else wr
        matrix[b][a] = None if math.isnan(wr) else (1.0 - wr)
        exceptions_total += summ["exceptions_left"] + summ["exceptions_right"]
    return matrix, pairs_summary, exceptions_total


def build_pairwise_full(state: dict, names: list[str]) -> dict[tuple[str, str], tuple[float, float]]:
    full: dict[tuple[str, str], tuple[float, float]] = {}
    for a, b in all_pairs(names):
        k = pair_key(a, b)
        games = state.get("roundrobin", {}).get(k, {}).get("games", [])
        summ = cell_summary(games)
        n = summ["decisive_and_draw"]
        we_a = summ["left_wins"] + 0.5 * summ["draws"]
        we_b = n - we_a
        full[(a, b)] = (we_a, n)
        full[(b, a)] = (we_b, n)
    return full


def write_dated_report(state: dict, games_target: int | None, mode_label: str) -> str:
    today = date.today().strftime("%Y%m%d")
    out_path = os.path.join(GATES_DIR, f"anchored_pool_{today}.json")
    matrix, pairs_summary, exceptions_total = build_matrix(state, ANCHOR_NAMES)

    n_pairs_total = len(all_pairs(ANCHOR_NAMES))
    n_pairs_complete = sum(
        1 for k, s in pairs_summary.items()
        if games_target is not None and s["n"] >= games_target
    ) if games_target is not None else None

    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode_label,
        "anchor_mu": ANCHOR_MU,
        "games_target_per_pair": games_target,
        "pairs_total": n_pairs_total,
        "pairs_complete": n_pairs_complete,
        "matrix_wr_row_vs_col": matrix,
        "pairs": pairs_summary,
        "exceptions_total": exceptions_total,
    }

    # state["candidate"] is FLAT: {pair_key -> {"left": cand_name, "right": anchor_name, "games": [...]}}
    # (same layout as state["roundrobin"], see record_result). Group by candidate (left) name here.
    cand_section = state.get("candidate", {})
    if cand_section:
        cand_payload: dict[str, dict] = {}
        for pk, cell in cand_section.items():
            cand_name = cell.get("left")
            anchor_name = cell.get("right")
            summ = cell_summary(cell.get("games", []))
            cand_payload.setdefault(cand_name, {})[anchor_name] = summ
        payload["candidates"] = cand_payload

    os.makedirs(GATES_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _bt_prob(mu_i: float, mu_j: float, s: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(mu_i - mu_j) / s))


def _weighted_xent(q: float, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(q * math.log(p) + (1 - q) * math.log(1 - p))


def fit_s(pairwise_full: dict, mu_known: dict, canonical_pairs: list[tuple[str, str]]) -> tuple[float, float]:
    def loss(s: float) -> float:
        total = 0.0
        for a, b in canonical_pairs:
            we, n = pairwise_full.get((a, b), (0.0, 0.0))
            if n <= 0:
                continue
            q = smoothed_wr(we, n)
            p = _bt_prob(mu_known[a], mu_known[b], s)
            total += n * _weighted_xent(q, p)
        return total

    grid = _log_grid(5.0, 4000.0, 300)
    losses = [loss(s) for s in grid]
    bi = min(range(len(grid)), key=lambda i: losses[i])
    lo = grid[max(0, bi - 1)]
    hi = grid[min(len(grid) - 1, bi + 1)]
    s_star = _golden_min(loss, lo, hi)
    return s_star, loss(s_star)


def estimate_mu_loo(pairwise_full: dict, mu_known: dict, k: str, others: list[str], s: float,
                     lo: float = 50.0, hi: float = 1300.0) -> float | None:
    def neg_ll(mu_k: float) -> float:
        total = 0.0
        for j in others:
            we, n = pairwise_full.get((k, j), (0.0, 0.0))
            if n <= 0:
                continue
            q = smoothed_wr(we, n)
            p = _bt_prob(mu_k, mu_known[j], s)
            total += n * _weighted_xent(q, p)
        return total

    if not any(pairwise_full.get((k, j), (0.0, 0.0))[1] > 0 for j in others):
        return None
    return _golden_min(neg_ll, lo, hi)


def find_inversions(names: list[str], mu_true: dict, mu_hat: dict) -> list[dict]:
    inv = []
    for a, b in itertools.combinations(names, 2):
        if mu_hat.get(a) is None or mu_hat.get(b) is None:
            continue
        d_true = mu_true[a] - mu_true[b]
        d_hat = mu_hat[a] - mu_hat[b]
        if d_true == 0:
            continue
        if (d_true > 0) != (d_hat > 0):
            inv.append({"a": a, "b": b, "mu_true_a": mu_true[a], "mu_true_b": mu_true[b],
                        "mu_hat_a": mu_hat[a], "mu_hat_b": mu_hat[b]})
    return inv


def chain_monotonic(mu_hat: dict, chain: list[str]) -> bool:
    vals = [mu_hat.get(c) for c in chain]
    if any(v is None for v in vals):
        return False
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def run_calibration(state: dict, out_path: str = CALIBRATION_PATH) -> dict:
    full = build_pairwise_full(state, ANCHOR_NAMES)

    def variant(names: list[str]) -> dict:
        pairs = all_pairs(names)
        s_star, loss_star = fit_s(full, ANCHOR_MU, pairs)
        mu_hat = {}
        for k in names:
            others = [n for n in names if n != k]
            mu_hat[k] = estimate_mu_loo(full, ANCHOR_MU, k, others, s_star)
        have_all = all(v is not None for v in mu_hat.values())
        sp = spearman([ANCHOR_MU[k] for k in names], [mu_hat[k] for k in names]) if have_all else float("nan")
        inv = find_inversions(names, ANCHOR_MU, mu_hat) if have_all else []
        chain_names = [c for c in CHAIN if c in names]
        mono = chain_monotonic(mu_hat, chain_names) if have_all else False
        verdict = "PASS" if (have_all and not math.isnan(sp) and sp >= 0.8 and len(inv) == 0) else "FAIL"
        return dict(anchors=names, s=s_star, loss=loss_star, mu_true={k: ANCHOR_MU[k] for k in names},
                    mu_hat=mu_hat, spearman=sp, inversions=inv, chain=chain_names,
                    chain_monotonic=mono, verdict=verdict)

    variant_all7 = variant(ANCHOR_NAMES)
    names_no002 = [n for n in ANCHOR_NAMES if n != WEAK_ANCHOR]
    variant_no002 = variant(names_no002)
    # v012_iono_heur is a known narrow Crustle-counter (local 0.70 vs live 499):
    # its local WRs against a Crustle-heavy pool are systematically inflated,
    # which is the prime suspect for the broken monotonicity above. Two extra
    # variants test that hypothesis: drop v012; drop v012 AND the stale v002.
    names_no012 = [n for n in ANCHOR_NAMES if n != "v012_iono_heur"]
    variant_no012 = variant(names_no012)
    names_no012_no002 = [n for n in ANCHOR_NAMES if n not in ("v012_iono_heur", WEAK_ANCHOR)]
    variant_no012_no002 = variant(names_no012_no002)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "anchor_mu": ANCHOR_MU,
        "weak_anchor_excluded_in_variant2": WEAK_ANCHOR,
        "variant_all7": variant_all7,
        "variant_no_v002": variant_no002,
        "variant_no_v012": variant_no012,
        "variant_no_v012_no_v002": variant_no012_no002,
    }

    # state["candidate"] is FLAT: {pair_key -> {"left": cand_name, "right": anchor_name, "games": [...]}}
    cand_section = state.get("candidate", {})
    if cand_section:
        cand_full = {}
        cand_names: set[str] = set()
        for pk, cell in cand_section.items():
            cand_name = cell.get("left")
            anchor_key = cell.get("right")
            cand_names.add(cand_name)
            summ = cell_summary(cell.get("games", []))
            n = summ["decisive_and_draw"]
            we = summ["left_wins"] + 0.5 * summ["draws"]
            cand_full[(cand_name, anchor_key)] = (we, n)
        candidates = {}
        for cand_name in cand_names:
            others7 = [a for a in ANCHOR_NAMES if (cand_name, a) in cand_full]
            others6 = [a for a in names_no002 if (cand_name, a) in cand_full]
            merged_full = dict(full)
            merged_full.update(cand_full)
            mu7 = estimate_mu_loo(merged_full, ANCHOR_MU, cand_name, others7, variant_all7["s"]) if others7 else None
            mu6 = estimate_mu_loo(merged_full, ANCHOR_MU, cand_name, others6, variant_no002["s"]) if others6 else None
            candidates[cand_name] = {"predicted_mu_all7": mu7, "predicted_mu_no_v002": mu6,
                                       "n_anchor_games": {a: cand_full[(cand_name, a)][1] for a in others7}}
        payload["candidates"] = candidates

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def print_calibration_report(payload: dict) -> None:
    for key in ("variant_all7", "variant_no_v002", "variant_no_v012", "variant_no_v012_no_v002"):
        if key not in payload:
            continue
        v = payload[key]
        print(f"\n=== {key} (anchors={len(v['anchors'])}) ===")
        print(f"  s = {v['s']:.2f}   loss = {v['loss']:.4f}   spearman = {v['spearman']:.4f}   "
              f"chain_monotonic = {v['chain_monotonic']}   verdict = {v['verdict']}")
        print(f"  {'anchor':22s} {'mu_true':>10s} {'mu_hat':>10s}")
        for k in v["anchors"]:
            mh = v["mu_hat"][k]
            mh_s = f"{mh:.1f}" if mh is not None else "N/A"
            print(f"  {k:22s} {v['mu_true'][k]:10.1f} {mh_s:>10s}")
        if v["inversions"]:
            print(f"  ИНВЕРСИИ ({len(v['inversions'])}):")
            for inv in v["inversions"]:
                print(f"    {inv['a']} (true={inv['mu_true_a']:.1f}, hat={inv['mu_hat_a']:.1f}) vs "
                      f"{inv['b']} (true={inv['mu_true_b']:.1f}, hat={inv['mu_hat_b']:.1f})")
        else:
            print("  инверсий нет")
    if "candidates" in payload:
        print("\n=== candidates ===")
        for name, c in payload["candidates"].items():
            print(f"  {name}: predicted_mu_all7={c['predicted_mu_all7']} predicted_mu_no_v002={c['predicted_mu_no_v002']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roundrobin", action="store_true", help="play the anchor x anchor round robin")
    ap.add_argument("--candidate", type=str, default=None, help="submissions dir (name or path): candidate vs each anchor")
    ap.add_argument("--anchors", type=str, default=None, help="comma-separated subset of anchor names (default: all 7)")
    ap.add_argument("--games-per-pair", type=int, default=100)
    ap.add_argument("--max-minutes", type=float, default=8.0, help="stop issuing new games after this wall-clock budget")
    ap.add_argument("--max-games-per-call", type=int, default=None, help="alt/extra cap on games STARTED this invocation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--state-path", type=str, default=STATE_PATH_DEFAULT)
    ap.add_argument("--calibrate", action="store_true", help="fit s + LOO validation from current state, no games played")
    ap.add_argument("--report-only", action="store_true", help="just (re)write the dated matrix report from current state")
    args = ap.parse_args()

    state = load_state(args.state_path)

    if args.calibrate:
        payload = run_calibration(state)
        print_calibration_report(payload)
        print(f"\nCalibration JSON: {CALIBRATION_PATH}")
        return 0

    if args.report_only:
        out = write_dated_report(state, args.games_per_pair, mode_label="report-only")
        print(f"Report: {out}")
        return 0

    if not args.roundrobin and not args.candidate:
        ap.error("укажи --roundrobin, --candidate <dir> или --calibrate/--report-only")

    deadline = time.time() + args.max_minutes * 60.0

    if args.roundrobin:
        names = ANCHOR_NAMES if args.anchors is None else [n.strip() for n in args.anchors.split(",") if n.strip()]
        for n in names:
            if n not in ANCHOR_MU:
                print(f"WARNING: {n!r} не в реестре якорей ANCHOR_MU (играем всё равно, но калибровка его не увидит)")
        pairs = all_pairs(names)
        print(f"[roundrobin] anchors={names}  pairs={len(pairs)}  games_target/pair={args.games_per_pair}  "
              f"workers={args.workers}  deadline_in={args.max_minutes:.1f}min")
        queue = build_task_queue(state, "roundrobin", pairs, args.games_per_pair, args.seed)
        print(f"[roundrobin] задач в очереди в этом запуске: {len(queue)}")
        n_done = run_queue(state, "roundrobin", queue, args.state_path, args.workers, deadline, args.max_games_per_call)
        print(f"[roundrobin] сыграно в этом запуске: {n_done}")

        print("\n--- прогресс по парам ---")
        for a, b in pairs:
            k = pair_key(a, b)
            summ = cell_summary(state.get("roundrobin", {}).get(k, {}).get("games", []))
            status = "COMPLETE" if summ["n"] >= args.games_per_pair else f"{summ['n']}/{args.games_per_pair}"
            wr = summ["wr_left"]
            wr_s = f"{wr:.3f}" if not math.isnan(wr) else "  n/a"
            print(f"  {k:45s} n={status:12s} wr_left={wr_s}  exc(L/R)={summ['exceptions_left']}/{summ['exceptions_right']}  "
                  f"draws={summ['draws']} aborted={summ['aborted']} infra_err={summ['infra_errors']}")

        out = write_dated_report(state, args.games_per_pair, mode_label="roundrobin")
        print(f"\nDated report: {out}")
        all_complete = all(
            cell_summary(state.get("roundrobin", {}).get(pair_key(a, b), {}).get("games", []))["n"] >= args.games_per_pair
            for a, b in pairs
        )
        print(f"ROUND ROBIN {'COMPLETE' if all_complete else 'INCOMPLETE — re-run to continue'}")

    if args.candidate:
        cand_dir = resolve_sub_dir(args.candidate)
        cand_name = os.path.basename(cand_dir.rstrip("/"))
        names = ANCHOR_NAMES if args.anchors is None else [n.strip() for n in args.anchors.split(",") if n.strip()]
        # Candidate is always "left" in every pair for a consistent state layout.
        pairs = [(cand_name, a) for a in names]
        # register candidate dir under its own name so resolve_sub_dir(cand_name) inside
        # build_task_queue still finds it even if it's not literally submissions/<cand_name>
        state.setdefault("_candidate_dirs", {})[cand_name] = cand_dir
        print(f"[candidate] {cand_name} ({cand_dir}) vs anchors={names}  games_target/pair={args.games_per_pair}  "
              f"workers={args.workers}  deadline_in={args.max_minutes:.1f}min")

        def _resolve_for_candidate(spec: str) -> str:
            if spec == cand_name:
                return cand_dir
            return resolve_sub_dir(spec)

        pending = []
        for a, b in pairs:
            k = pair_key(a, b)
            have = len(state.get("candidate", {}).get(k, {}).get("games", []))
            need = max(0, args.games_per_pair - have)
            pending.append([a, b, k, have, need, 0])
        queue = []
        progressed = True
        while progressed:
            progressed = False
            for row in pending:
                a, b, k, have, need, ctr = row
                if ctr < need:
                    game_idx = have + ctr
                    left_first = (game_idx % 2 == 0)
                    seed = args.seed + (hash((k, game_idx)) % 1_000_000)
                    queue.append(dict(
                        pair_key=k, left_name=a, right_name=b,
                        left_dir=_resolve_for_candidate(a), right_dir=_resolve_for_candidate(b),
                        left_first=left_first, game_idx=game_idx, seed=seed,
                    ))
                    row[5] += 1
                    progressed = True

        print(f"[candidate] задач в очереди в этом запуске: {len(queue)}")
        n_done = run_queue(state, "candidate", queue, args.state_path, args.workers, deadline, args.max_games_per_call)
        print(f"[candidate] сыграно в этом запуске: {n_done}")

        print("\n--- прогресс кандидата по якорям ---")
        for a, b in pairs:
            k = pair_key(a, b)
            summ = cell_summary(state.get("candidate", {}).get(k, {}).get("games", []))
            status = "COMPLETE" if summ["n"] >= args.games_per_pair else f"{summ['n']}/{args.games_per_pair}"
            wr = summ["wr_left"]
            wr_s = f"{wr:.3f}" if not math.isnan(wr) else "  n/a"
            print(f"  {k:45s} n={status:12s} wr_cand={wr_s}  exc(cand/anchor)={summ['exceptions_left']}/{summ['exceptions_right']}")

        out = write_dated_report(state, args.games_per_pair, mode_label="candidate")
        print(f"\nDated report: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
