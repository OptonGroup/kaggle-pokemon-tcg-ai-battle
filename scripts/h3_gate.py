"""h3_gate.py — H3-4 Phase-1 гейт (50g пилот) для PIMCAgentV2.

PLAN.md §H3-4: полный гейт по 3 плечам с BOEВЫМ bank_total_sec=600.

Три плеча (Phase-1 = 50g на ячейку, чередование сторон):
  (a) MIRROR: PIMCAgentV2(Crustle) vs v009a_crustle (flat-IL prior). 50g.
      Gate: WR≥0.55 И Wilson_lo>0.50.
  (b) FIELD-ДЕЛЬТА (KILL-плечо):
      Поле F = {v017_alakazam_heur, v018_grimmsnarl,
               meta:alakazam_aidy, meta:grimmsnarl_kazuki, meta:cubchoo_mtn}.
      Для КАЖДОГО opp: 50g PIMC vs opp И 50g v009a vs opp.
      Δ = WR(PIMC vs opp) − WR(v009a vs opp).
      Gate: Δ≥+0.05 для КАЖДОГО opp, ни одного отрицательного.
      KILL: любой Δ<0.
  (c) PREDICTED-μ: PIMCAgentV2 vs 5 якорей {v015, v001, v018, v009a, v007} по 50g.
      BT-фит μ̂ со скаляром s=413.3 (variant_no_v012_no_v002).
      Gate: μ̂ > 783.
      RED-FLAG: WR vs v001_random < 0.90 → СТОП.

Multiprocessing spawn 4 воркера. PIMCAgentV2 не пиклится — воркер создаёт
всё внутри себя (v009a prior + PIMCAgentV2 + _lazy_init). В task только bare
данные (примитивы). Resumable state JSON.

Запуск:
    # sanity-check (1 игра mirror)
    .venv/bin/python scripts/h3_gate.py --games 1 --arms mirror

    # полный Phase-1 (все 3 плеча, 50g)
    .venv/bin/python scripts/h3_gate.py --games 50 --arms a,b,c --workers 4

    # продолжить прерванный прогон
    .venv/bin/python scripts/h3_gate.py --games 50 --arms a,b,c --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Нативный движок (macOS arm64). Threading НЕБЕЗОПАСЕН — только multiprocessing spawn.
from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

from exit.pimc_agent import PIMCAgentV2, CRUSTLE_DECK_APPROX  # noqa: E402

# cg-алиас: submissions/*/main.py могут делать `from cg.api import ...` —
# на macOS это бы упало (Linux .so), подменяем на нативные модули.
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)
MAX_MOVES_PER_GAME = 2000

GATES_DIR = os.path.join(ROOT, "results", "gates")
STATE_PATH = os.path.join(GATES_DIR, "h3_gate_state.json")
REPORT_PATH = os.path.join(GATES_DIR, "h3_gate_phase1.json")

# BOEВОЙ bank (не smoke-овый 90). max_decision_budget=3.0.
BANK_KW = dict(
    bank_total_sec=600.0,
    bank_safety=0.6,
    est_total_decisions=40,
    min_decision_budget=0.5,
    max_decision_budget=3.0,
)
EPSILON = 0.3

# Якоря live-μ (frozen 2026-07-11). v012 и v002 ИСКЛЮЧЕНЫ.
ANCHOR_MU: dict[str, float] = {
    "v015_exit_pilot": 317.7,
    "v001_random": 483.8,
    "v018_grimmsnarl": 484.1,
    "v009a_crustle": 724.1,
    "v007_crustle": 733.1,
}
S_BT = 413.3  # BT-скаляр variant_no_v012_no_v002 (из anchored_calibration.json)

# Поле оппонентов для плеча (b)
FIELD_OPPS: list[str] = [
    "v017_alakazam_heur",
    "v018_grimmsnarl",
    "meta:alakazam_aidy",
    "meta:grimmsnarl_kazuki",
    "meta:cubchoo_mtn",
]

# Якоря для плеча (c) — 5 якорей
ANCHOR_OPPS: list[str] = ["v015_exit_pilot", "v001_random", "v018_grimmsnarl",
                            "v009a_crustle", "v007_crustle"]

# ---------------------------------------------------------------------------
# Мета-дек greedy-attack пилот (БЕЗ lgb, БЕЗ cg.api top-level импорта)
# ---------------------------------------------------------------------------

_ATTACK_DAMAGE_MAP: dict[int, int] | None = None


def _get_attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_MAP
    if _ATTACK_DAMAGE_MAP is None:
        try:
            from cg_native.api import all_attack
        except ImportError:
            from cg.api import all_attack  # type: ignore[no-redef]
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def _opt_get(o, key: str, default=None):
    """Get attribute from dict or object."""
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


# OptionType int-константы (см. pimc_agent.py / cg_native/api.py)
_OT_YES = 1
_OT_CARD = 3
_OT_PLAY = 7
_OT_ATTACH = 8
_OT_EVOLVE = 9
_OT_ABILITY = 10
_OT_RETREAT = 12
_OT_ATTACK = 13
_OT_END = 14
_ST_YES_NO = 9
_ST_COUNT = 8
_SC_MAIN = 0
_GREEDY_PRIORITY: tuple[int, ...] = (
    _OT_ATTACK, _OT_ABILITY, _OT_EVOLVE, _OT_PLAY, _OT_ATTACH, _OT_RETREAT, _OT_CARD, _OT_END,
)


def greedy_meta_agent(obs_dict: dict) -> list[int]:
    """Generic greedy-attack пилот для мета-дек-оппонентов.

    Без lgb, без cg.api top-level импорта. all_attack() вызывается лениво при
    первом вызове через cg-алиас. Атака с макс. уроном; иначе приоритет по типу
    действия; иначе первый легальный ход. И PIMC, и v009a играют ПРОТИВ
    ИДЕНТИЧНОГО пилота на той же мета-деке.
    """
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
    if sel_type == _ST_COUNT:
        best = max(range(n), key=lambda i: (_opt_get(opts[i], "number", 0) or 0))
        return [best]
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    context = sel.get("context")
    if context == _SC_MAIN:
        atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
            return [best]
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


# ---------------------------------------------------------------------------
# Загрузка модулей сабмитов (fresh per call — stateful-audit, как smoke)
# ---------------------------------------------------------------------------

_LOAD_COUNTER = 0


def _load_submission_module(sub_dir_name: str):
    """Загружает submissions/<sub_dir_name>/main.py свежим instance'ом модуля.
    chdir в директорию сабмита (некоторые агенты читают файлы из CWD).
    """
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_h3_gate_{sub_dir_name}_{_LOAD_COUNTER}"
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(ids)}")
    return ids


def _resolve_opp_deck(sub_dir_name: str) -> list[int]:
    """Дек оппонента-сабмита: mod.my_deck или deck.csv fallback."""
    return _load_deck_csv(os.path.join(ROOT, "submissions", sub_dir_name, "deck.csv"))


def _load_meta_deck(deck_name: str) -> list[int]:
    """Дек мета-оппонента из decks/meta/<deck_name>.csv."""
    path = os.path.join(ROOT, "decks", "meta", f"{deck_name}.csv")
    return _load_deck_csv(path)


def _meta_deck_name(opp_spec: str) -> str:
    """meta:alakazam_aidy -> alakazam_aidy"""
    assert opp_spec.startswith("meta:"), f"not a meta spec: {opp_spec}"
    return opp_spec[5:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def smoothed_wr(we: float, n: float) -> float:
    """Laplace-smoothed win fraction (+1/+2) только на 0%/100% краях."""
    if n <= 0:
        return 0.5
    wr = we / n
    if wr <= 0.0 or wr >= 1.0:
        return (we + 1.0) / (n + 2.0)
    return wr


def _bt_prob(mu_i: float, mu_j: float, s: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(mu_i - mu_j) / s))


def _weighted_xent(q: float, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(q * math.log(p) + (1 - q) * math.log(1 - p))


def _golden_min(f, a: float, b: float, tol: float = 1e-3, max_iter: int = 200) -> float:
    """1-D unimodal minimizer (golden section), no scipy."""
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


def estimate_mu_hat(wr_data: dict[str, tuple[float, int]], anchor_mu: dict[str, float],
                    s: float, lo: float = 50.0, hi: float = 1300.0) -> float:
    """BT-фит μ̂: минимизация правдоподобия по μ при фиксированном s.

    wr_data: {anchor_name: (win_equivalent, n_decisive)}
    anchor_mu: {anchor_name: live_mu}
    s: BT-скаляр (413.3)
    """
    def neg_ll(mu: float) -> float:
        total = 0.0
        for anchor, (we, n) in wr_data.items():
            if n <= 0:
                continue
            q = smoothed_wr(we, n)
            p = _bt_prob(mu, anchor_mu[anchor], s)
            total += n * _weighted_xent(q, p)
        return total
    return _golden_min(neg_ll, lo, hi)


# ---------------------------------------------------------------------------
# Worker: _play_one_game — ВСЁ создаётся внутри воркера (spawn-safe)
# ---------------------------------------------------------------------------

def _play_one_game(task: dict) -> dict:
    cell_id = task["cell_id"]
    arm = task["arm"]
    player = task["player"]
    opp_spec = task["opp_spec"]
    our_first = task["our_first"]
    seed = task["seed"]
    game_idx = task["game_idx"]

    import random
    random.seed(seed)

    # --- Создание игрока ---
    pimc = None
    worlds_before = 0

    if player == "pimc":
        prior_mod = _load_submission_module("v009a_crustle")
        pimc = PIMCAgentV2(
            deck=list(CRUSTLE_DECK),
            prior_agent=prior_mod.agent,
            opponent_model="auto",
            epsilon=EPSILON,
            rng_seed=seed,
            **BANK_KW,
        )
        pimc._lazy_init()
        worlds_before = len(pimc._pimc_mod._stats.get("worlds_per_decision", []))
        player_agent = pimc.agent
        player_deck = list(CRUSTLE_DECK)
    else:  # "v009a"
        v009a_mod = _load_submission_module("v009a_crustle")
        player_agent = v009a_mod.agent
        player_deck = list(getattr(v009a_mod, "my_deck", None) or _resolve_opp_deck("v009a_crustle"))

    # --- Создание оппонента ---
    if opp_spec.startswith("meta:"):
        deck_name = _meta_deck_name(opp_spec)
        opp_deck = _load_meta_deck(deck_name)
        opp_agent = greedy_meta_agent
    else:
        opp_mod = _load_submission_module(opp_spec)
        opp_agent = opp_mod.agent
        opp_deck = list(getattr(opp_mod, "my_deck", None) or _resolve_opp_deck(opp_spec))

    # --- Расстановка сторон ---
    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_agent, opp_agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = opp_agent, player_agent

    # --- Игра ---
    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(
            cell_id=cell_id, arm=arm, player=player, opp_spec=opp_spec,
            game_idx=game_idx, our_first=our_first, seed=seed,
            outcome="crash", error=f"battle_start: {getattr(start, 'errorType', None)}",
            moves=0, n_illegal=0, n_exceptions=1, exc_side=None,
            decisions_total=None, n_search_decisions=None, n_cheap_decisions=None,
            search_share=None, ms_per_decision_median=0.0, ms_per_decision_max=0.0,
            bank_spent_sec=None, wall_clock_sec=None,
            rollouts_per_search_decision_median=None, rollout_degradation={},
            locked_archetype=None, wall_time_game_sec=0.0,
        )

    move_count = 0
    our_decision_times: list[float] = []
    n_illegal = 0
    n_exceptions = 0
    winner = None
    exc_side = None

    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break

            your_index = obs.current.yourIndex if obs.current else 0
            is_our_turn = (your_index == 0) == our_first
            agent_fn = agent0 if your_index == 0 else agent1

            sel = obs_dict.get("select")
            t0 = time.perf_counter()
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                break
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if is_our_turn:
                our_decision_times.append(dt_ms)

            if not _is_legal(choice, sel):
                n_illegal += 1
                break

            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    # --- Исход ---
    our_side = 0 if our_first else 1
    if n_exceptions > 0:
        outcome = "exception"
    elif n_illegal > 0:
        outcome = "illegal"
    elif winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    result: dict = dict(
        cell_id=cell_id, arm=arm, player=player, opp_spec=opp_spec,
        game_idx=game_idx, our_first=our_first, seed=seed,
        outcome=outcome, exc_side=exc_side, winner=winner, moves=move_count,
        n_illegal=n_illegal, n_exceptions=n_exceptions,
        ms_per_decision_median=_median(our_decision_times),
        ms_per_decision_max=(max(our_decision_times) if our_decision_times else 0.0),
    )

    if pimc is not None:
        worlds_after = len(pimc._pimc_mod._stats.get("worlds_per_decision", []))
        worlds_this_game = pimc._pimc_mod._stats["worlds_per_decision"][worlds_before:worlds_after]
        bank = pimc.bank_stats
        dt = bank["decisions_made"]
        result.update(
            decisions_total=dt,
            n_search_decisions=bank["n_search_decisions"],
            n_cheap_decisions=bank["n_cheap_decisions"],
            search_share=(bank["n_search_decisions"] / dt) if dt else None,
            rollouts_per_search_decision_median=_median([float(w) for w in worlds_this_game]),
            bank_spent_sec=bank["spent_sec"],
            wall_clock_sec=bank["wall_clock_sec"],
            rollout_degradation=dict(pimc.rollout_degradation),
            locked_archetype=pimc._locked_archetype,
        )
    else:
        result.update(
            decisions_total=None, n_search_decisions=None, n_cheap_decisions=None,
            search_share=None, rollouts_per_search_decision_median=None,
            bank_spent_sec=None, wall_clock_sec=None,
            rollout_degradation={}, locked_archetype=None,
        )

    return result


# ---------------------------------------------------------------------------
# State persistence (resumable)
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cells": {}}


def save_state(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def cell_id_for(player: str, opp_spec: str) -> str:
    return f"{player}__vs__{opp_spec}"


def record_game(state: dict, cell_id: str, cell_meta: dict, result: dict) -> None:
    cells = state.setdefault("cells", {})
    if cell_id not in cells:
        cells[cell_id] = {**cell_meta, "games": []}
    cells[cell_id]["games"].append(result)


# ---------------------------------------------------------------------------
# Task queue: все ячейки всех плеч, interleaved (round-robin по ячейкам)
# ---------------------------------------------------------------------------

# Определение ячеек: (cell_id, arm, player, opp_spec, arms_list)
def build_cell_specs(arms: set[str]) -> list[dict]:
    """Возвращает список ячеек для выбранных плеч. Ячейки разделяются между
    плечами где это возможно (pimc vs v009a = arm a + arm c; pimc vs v018 = arm b + arm c)."""
    cells: dict[str, dict] = {}

    def add(cell_id: str, arm: str, player: str, opp_spec: str):
        if cell_id not in cells:
            cells[cell_id] = dict(cell_id=cell_id, player=player, opp_spec=opp_spec, arms=set())
        cells[cell_id]["arms"].add(arm)

    if "a" in arms or "c" in arms:
        # pimc vs v009a (arm a + arm c)
        add("pimc__vs__v009a_crustle", "a", "pimc", "v009a_crustle")
        if "c" in arms:
            add("pimc__vs__v009a_crustle", "c", "pimc", "v009a_crustle")

    if "b" in arms:
        for opp in FIELD_OPPS:
            add(f"pimc__vs__{opp}", "b", "pimc", opp)
            add(f"v009a__vs__{opp}", "b", "v009a", opp)

    if "c" in arms:
        for anchor in ANCHOR_OPPS:
            add(f"pimc__vs__{anchor}", "c", "pimc", anchor)
        # v009a и v018 уже добавлены из arm a / arm b — добавим arm c если их нет
        if "b" not in arms:
            # Если arm b не выбран, pimc vs v018 нужна только для arm c
            pass  # уже добавлено выше

    # Конвертируем sets в списки для JSON-сериализации
    for cs in cells.values():
        cs["arms"] = sorted(cs["arms"])

    return list(cells.values())


def build_task_queue(state: dict, cell_specs: list[dict], games_target: int,
                      base_seed: int) -> list[dict]:
    """Interleaved очередь: round-robin по ячейкам, чтобы partial run затронул все."""
    pending = []
    for cs in cell_specs:
        cid = cs["cell_id"]
        have = len(state.get("cells", {}).get(cid, {}).get("games", []))
        need = max(0, games_target - have)
        pending.append([cs, have, need, 0])

    queue: list[dict] = []
    progressed = True
    while progressed:
        progressed = False
        for row in pending:
            cs, have, need, ctr = row
            if ctr < need:
                game_idx = have + ctr
                our_first = (game_idx % 2 == 0)
                seed = base_seed + (hash((cs["cell_id"], game_idx)) % 1_000_000)
                queue.append(dict(
                    cell_id=cs["cell_id"],
                    arm="|".join(sorted(cs["arms"])),
                    player=cs["player"],
                    opp_spec=cs["opp_spec"],
                    our_first=our_first,
                    seed=seed,
                    game_idx=game_idx,
                ))
                row[3] += 1
                progressed = True
    return queue


# ---------------------------------------------------------------------------
# Run queue (multiprocessing spawn, 4 workers, resumable)
# ---------------------------------------------------------------------------

def run_queue(state: dict, queue: list[dict], cell_specs: list[dict],
              state_path: str, workers: int, deadline: float) -> tuple[int, int]:
    """Запускает задачи через spawn ProcessPoolExecutor, сохраняет state после
    каждой игры. Возвращает (n_finished, n_illegal_total, n_exc_total)."""
    if not queue:
        return 0, 0, 0

    cell_meta_map = {cs["cell_id"]: cs for cs in cell_specs}
    ctx = multiprocessing.get_context("spawn")
    q_iter = iter(queue)
    n_started = 0
    n_finished = 0
    total_illegal = 0
    total_exc = 0

    def _can_submit_more() -> bool:
        return time.time() < deadline

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
                    res = dict(
                        cell_id=t["cell_id"], arm=t["arm"], player=t["player"],
                        opp_spec=t["opp_spec"], game_idx=t["game_idx"],
                        our_first=t["our_first"], seed=t["seed"],
                        outcome="crash", error=repr(e), moves=0,
                        n_illegal=0, n_exceptions=1, exc_side=None,
                        decisions_total=None, n_search_decisions=None,
                        n_cheap_decisions=None, search_share=None,
                        ms_per_decision_median=0.0, ms_per_decision_max=0.0,
                        bank_spent_sec=None, wall_clock_sec=None,
                        rollouts_per_search_decision_median=None,
                        rollout_degradation={}, locked_archetype=None,
                        wall_time_game_sec=0.0,
                    )

                cs = cell_meta_map[t["cell_id"]]
                record_game(state, t["cell_id"], cs, res)
                save_state(state, state_path)
                n_finished += 1
                total_illegal += res.get("n_illegal", 0)
                total_exc += res.get("n_exceptions", 0)

                out = res.get("outcome", "?")
                tag = "" if res.get("outcome") not in ("crash", "exception", "illegal") else f" [{res.get('error', '')}]"
                print(f"  [{t['cell_id']}] game {t['game_idx']:3d} first={'Y' if t['our_first'] else 'N'} "
                      f"outcome={out:9s} moves={res.get('moves', 0):4d} "
                      f"ms/dec={res.get('ms_per_decision_median', 0):.0f} "
                      f"bank={res.get('bank_spent_sec') or 0:.1f}s{tag}", flush=True)

                if _can_submit_more():
                    try:
                        nt = next(q_iter)
                        futures[ex.submit(_play_one_game, nt)] = nt
                        n_started += 1
                    except StopIteration:
                        pass

    return n_finished, total_illegal, total_exc


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def cell_wr(games: list[dict]) -> tuple[float, int, int, int, float, float, float]:
    """Возвращает (wr, wins, losses, draws, wilson_lo, wilson_hi, n_decisive) для ячейки."""
    wins = sum(1 for g in games if g.get("outcome") == "our_win")
    losses = sum(1 for g in games if g.get("outcome") == "opp_win")
    draws = sum(1 for g in games if g.get("outcome") == "draw")
    n = wins + losses + draws
    we = wins + 0.5 * draws
    wr = we / n if n > 0 else float("nan")
    lo, hi = wilson_ci(we, n) if n > 0 else (float("nan"), float("nan"))
    return wr, wins, losses, draws, lo, hi, n


def build_report(state: dict, games_target: int) -> dict:
    cells = state.get("cells", {})

    def get_games(cid: str) -> list[dict]:
        return cells.get(cid, {}).get("games", [])

    # --- Arm (a): MIRROR ---
    a_games = get_games("pimc__vs__v009a_crustle")
    a_wr, a_w, a_l, a_d, a_lo, a_hi, a_n = cell_wr(a_games)
    a_pimc_games = [g for g in a_games if g.get("outcome") not in ("crash",)]
    a_ms_meds = [g["ms_per_decision_median"] for g in a_pimc_games if g.get("ms_per_decision_median")]
    a_bank_spents = [g["bank_spent_sec"] for g in a_pimc_games if g.get("bank_spent_sec") is not None]
    a_search_shares = [g["search_share"] for g in a_pimc_games if g.get("search_share") is not None]
    a_rollouts_meds = [g["rollouts_per_search_decision_median"] for g in a_pimc_games
                       if g.get("rollouts_per_search_decision_median") is not None]
    a_pass = (a_wr >= 0.55 and a_lo > 0.50) if a_n > 0 else False
    arm_a = dict(
        n=a_n, wins=a_w, losses=a_l, draws=a_d, WR=a_wr,
        wilson_lo=a_lo, wilson_hi=a_hi, gate_pass=a_pass,
        ms_per_decision_median=_median(a_ms_meds),
        bank_spent_max_sec=max(a_bank_spents) if a_bank_spents else 0.0,
        search_share=_median(a_search_shares) if a_search_shares else None,
        rollouts_per_decision_median=_median(a_rollouts_meds) if a_rollouts_meds else 0.0,
    )

    # --- Arm (b): FIELD-ДЕЛЬТА ---
    arm_b_table: list[dict] = []
    for opp in FIELD_OPPS:
        pimc_cid = f"pimc__vs__{opp}"
        v009a_cid = f"v009a__vs__{opp}"
        p_games = get_games(pimc_cid)
        v_games = get_games(v009a_cid)
        p_wr, _, _, _, _, _, p_n = cell_wr(p_games)
        v_wr, _, _, _, _, _, v_n = cell_wr(v_games)
        delta = p_wr - v_wr if not (math.isnan(p_wr) or math.isnan(v_wr)) else float("nan")
        arm_b_table.append(dict(
            opp=opp, n_pimc=p_n, WR_pimc=p_wr, n_v009a=v_n, WR_v009a=v_wr,
            delta=delta, delta_pass=(delta >= 0.05) if not math.isnan(delta) else False,
            delta_negative=(delta < 0) if not math.isnan(delta) else False,
        ))
    b_all_pass = all(row["delta_pass"] for row in arm_b_table) if arm_b_table else False
    b_no_negative = all(not row["delta_negative"] for row in arm_b_table) if arm_b_table else False
    b_kill = any(row["delta_negative"] for row in arm_b_table)

    # --- Arm (c): PREDICTED-μ ---
    wr_data: dict[str, tuple[float, int]] = {}
    arm_c_per_anchor: list[dict] = []
    for anchor in ANCHOR_OPPS:
        cid = f"pimc__vs__{anchor}"
        games = get_games(cid)
        wr, w, l, d, lo, hi, n = cell_wr(games)
        we = w + 0.5 * d
        wr_data[anchor] = (we, n)
        arm_c_per_anchor.append(dict(anchor=anchor, n=n, WR=wr, wilson_lo=lo, wilson_hi=hi,
                                       live_mu=ANCHOR_MU.get(anchor)))

    # RED-FLAG: WR vs v001_random
    v001_wr_data = wr_data.get("v001_random", (0.0, 0))
    v001_wr = v001_wr_data[0] / v001_wr_data[1] if v001_wr_data[1] > 0 else float("nan")
    red_flag_triggered = (not math.isnan(v001_wr) and v001_wr < 0.90)

    # BT-фит μ̂
    mu_hat = None
    if any(n > 0 for _, n in wr_data.values()):
        mu_hat = estimate_mu_hat(wr_data, ANCHOR_MU, S_BT)
    c_pass = (mu_hat is not None and mu_hat > 783)
    arm_c = dict(
        per_anchor=arm_c_per_anchor,
        mu_hat=mu_hat,
        gate_pass=c_pass,
        s_used=S_BT,
    )

    # --- Verdict ---
    if b_kill:
        verdict = "KILL_PIMC"
    elif red_flag_triggered:
        verdict = "RED_FLAG_STOP"
    elif a_pass and b_all_pass and b_no_negative and c_pass:
        verdict = "GATE_PASS"
    elif (a_wr >= 0.52 if a_n > 0 else False) and b_no_negative and (mu_hat is not None and mu_hat > 733):
        verdict = "PROMISING_EXPAND_TO_300G"
    else:
        verdict = "INCONCLUSIVE"

    # --- Timing / stats ---
    all_pimc_games = []
    for cid, cell in cells.items():
        for g in cell.get("games", []):
            if g.get("player") == "pimc" and g.get("outcome") not in ("crash",):
                all_pimc_games.append(g)
    all_ms_meds = [g["ms_per_decision_median"] for g in all_pimc_games if g.get("ms_per_decision_median")]
    all_bank_spents = [g["bank_spent_sec"] for g in all_pimc_games if g.get("bank_spent_sec") is not None]
    all_degrad: dict = {}
    for g in all_pimc_games:
        for k, v in (g.get("rollout_degradation") or {}).items():
            all_degrad[k] = all_degrad.get(k, 0) + v

    total_illegal = sum(g.get("n_illegal", 0) for cell in cells.values() for g in cell.get("games", []))
    total_exc = sum(g.get("n_exceptions", 0) for cell in cells.values() for g in cell.get("games", []))

    report = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        games_target=games_target,
        arm_a_mirror=arm_a,
        arm_b_field=dict(
            table=arm_b_table,
            field_gate_pass=b_all_pass and b_no_negative,
            kill=b_kill,
        ),
        arm_c_predicted_mu=arm_c,
        red_flag_v001=dict(WR_vs_v001=v001_wr, triggered=red_flag_triggered),
        verdict=verdict,
        timing=dict(
            total_illegal=total_illegal,
            total_exceptions=total_exc,
            pimc_games_total=len(all_pimc_games),
            avg_ms_per_decision=_median(all_ms_meds) if all_ms_meds else 0.0,
            bank_spent_max_sec=max(all_bank_spents) if all_bank_spents else 0.0,
            bank_spent_median_sec=_median(all_bank_spents) if all_bank_spents else 0.0,
            rollout_degradation_total=all_degrad,
        ),
        field_composition=dict(
            pilot="greedy_meta_agent (scripts/h3_gate.py)",
            meta_decks_identical_for_pimc_and_v009a=True,
            note="PIMC и v009a играют против ИДЕНТИЧНОГО greedy-пилота на той же мета-деке",
        ),
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=50, help="партий на ячейку")
    ap.add_argument("--arms", type=str, default="a,b,c", help="плечи: a,b,c или mirror (alias a)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=240.0, help="дедлайн прогона (мин)")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--state-path", type=str, default=STATE_PATH)
    ap.add_argument("--report-path", type=str, default=REPORT_PATH)
    ap.add_argument("--report-only", action="store_true", help="только отчёт из текущего state")
    args = ap.parse_args()

    # Парсинг плеч
    arms_str = args.arms.lower().strip()
    if arms_str == "mirror":
        arms = {"a"}
    else:
        arms = set(a.strip() for a in arms_str.split(",") if a.strip())
        if "mirror" in arms:
            arms.discard("mirror")
            arms.add("a")
    for a in arms:
        assert a in ("a", "b", "c"), f"unknown arm {a!r}"

    state = load_state(args.state_path)

    if args.report_only:
        report = build_report(state, args.games)
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w") as f:
            json.dump(report, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
        print(json.dumps(report, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x))
        return 0

    cell_specs = build_cell_specs(arms)
    queue = build_task_queue(state, cell_specs, args.games, args.seed)
    print(f"[h3_gate] arms={sorted(arms)} cells={len(cell_specs)} tasks={len(queue)} "
          f"workers={args.workers} max_minutes={args.max_minutes}", flush=True)
    for cs in cell_specs:
        cid = cs["cell_id"]
        have = len(state.get("cells", {}).get(cid, {}).get("games", []))
        print(f"  {cid:45s} have={have:3d}/{args.games} arms={sorted(cs['arms'])}", flush=True)

    if not queue:
        print("[h3_gate] очередь пуста — все ячейки уже заполнены")
    else:
        deadline = time.time() + args.max_minutes * 60.0
        t0 = time.time()
        n_done, n_illegal, n_exc = run_queue(state, queue, cell_specs, args.state_path,
                                              args.workers, deadline)
        elapsed = time.time() - t0
        print(f"\n[h3_gate] сыграно: {n_done}, illegal: {n_illegal}, exceptions: {n_exc}, "
              f"время: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # --- Отчёт ---
    report = build_report(state, args.games)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\n=== ОТЧЁТ ===")
    print(json.dumps(report, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x))
    print(f"\nReport JSON: {args.report_path}")
    print(f"State JSON: {args.state_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())