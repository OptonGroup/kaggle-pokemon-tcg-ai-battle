"""PIMC Hybrid агент: эвристика v012 + PIMC только для ключевых MAIN решений.

Архитектура:
- Для всех решений КРОМЕ MAIN select: используем эвристику v012 напрямую.
- Для MAIN select: если есть реальный выбор (ATTACK vs не-ATTACK, или несколько атак),
  используем PIMC чтобы определить — атаковать сейчас или подождать/построить доску.
- PIMC rollout: эвристическая policy (простые правила) вместо случайной.

Смысл: эвристика v012 хорошо знает КАК делать каждое действие,
но PIMC может определить КОГДА выгоднее атаковать vs накапливать энергию.
"""

from __future__ import annotations

import os
import sys
import random
import time
import itertools
from collections import defaultdict

# ---------------------------------------------------------------------------
# Загружаем Iono эвристику v012 (основная policy)
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_v012_mod = None


def _load_v012():
    global _v012_mod
    if _v012_mod is not None:
        return
    v12_path = os.path.join(_REPO, "submissions/v012_ref/main.py")
    if not os.path.exists(v12_path):
        # Попробуем v012_iono_heur
        v12_path = os.path.join(_REPO, "submissions/v012_iono_heur/main.py")
    if os.path.exists(v12_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_v012_main", v12_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = v12_path
        v12_dir = os.path.dirname(v12_path)
        if v12_dir not in sys.path:
            sys.path.insert(0, v12_dir)
        spec.loader.exec_module(mod)
        _v012_mod = mod
    else:
        _v012_mod = None


from cg.api import (
    to_observation_class, search_begin, search_step, search_end, search_release,
    Observation, SelectContext, OptionType, SelectType,
)

# ---------------------------------------------------------------------------
# Iono дека
# ---------------------------------------------------------------------------
IONO_DECK: list[int] = [
    265, 265, 265,
    268, 268, 268,
    269, 269, 269,
    270, 270, 270,
    271, 271, 271,
    1233, 1233, 1233, 1233,
    1227, 1227, 1227, 1227,
    1086, 1086, 1086,
    1121, 1121, 1121,
    1254, 1254, 1254,
    1097, 1097,
    1152, 1152,
    1110,
    1118,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4,
]

TIME_BUDGET_SEC: float = 2.0
N_WORLDS_MIN: int = 4
MAX_ROLLOUT_DEPTH: int = 100
PIMC_ENABLED: bool = True  # можно отключить для бенчмарка чистой эвристики

_stats = {
    "rollouts_total": 0,
    "decisions_pimc": 0,
    "decisions_heuristic": 0,
    "decisions_trivial": 0,
}


def get_stats():
    return dict(_stats)


def _deck_ids():
    return list(IONO_DECK)


def _heuristic_action(obs_dict: dict) -> list[int]:
    """Вызываем v012 эвристику или fallback."""
    _load_v012()
    if _v012_mod is not None:
        try:
            return _v012_mod.agent(obs_dict)
        except Exception:
            pass
    # Простой fallback
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return _deck_ids()
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    min_c = int(sel.get("minCount") or 1)
    max_c = min(int(sel.get("maxCount") or 1), n)
    k = max(1, min_c)
    return list(range(min(k, n)))


# ---------------------------------------------------------------------------
# Скрытое состояние оппонента
# ---------------------------------------------------------------------------

def _sample_hidden(known_ids, full_deck, count):
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
    return random.sample(pool, min(count, len(pool)))


def _opp_known(obs, opp_idx):
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


def _build_search_inputs(obs, your_index):
    my = obs.current.players[your_index]
    opp_idx = 1 - your_index
    opp = obs.current.players[opp_idx]

    ydc = my.deckCount
    your_deck = (list(IONO_DECK) + [4] * 10)[:max(ydc, 0)]
    your_prize = [p.id if p is not None else 4 for p in (my.prize or [])]

    opp_full = list(IONO_DECK)
    opp_kn = _opp_known(obs, opp_idx)
    opp_deck = _sample_hidden(opp_kn, opp_full, opp.deckCount)
    opp_prize = []
    for p in (opp.prize or []):
        if p is not None:
            opp_prize.append(p.id)
        else:
            pool = [x for x in opp_full if x not in opp_prize]
            opp_prize.append(random.choice(pool) if pool else 4)
    opp_hand = _sample_hidden(opp_kn, opp_full, opp.handCount)

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand


def _opp_active_ids(obs, opp_idx):
    opp = obs.current.players[opp_idx]
    if opp.active and opp.active[0] is None:
        return [265]
    return []


# ---------------------------------------------------------------------------
# Эвристический rollout (без v012 — только правила Iono)
# ---------------------------------------------------------------------------

def _simple_rollout_choice(obs: Observation) -> list[int]:
    """Простые правила для rollout: ATTACK > ABILITY > EVOLVE > PLAY > ATTACH > END."""
    sel = obs.select
    if sel is None:
        return []
    opts = sel.option
    n = len(opts)
    if n == 0:
        return []
    min_c = sel.minCount
    max_c = min(sel.maxCount, n)

    if sel.type == SelectType.YES_NO:
        for i, o in enumerate(opts):
            if o.type == OptionType.YES:
                return [i]
        return [0]
    if sel.type == SelectType.COUNT:
        best = max(range(n), key=lambda i: (opts[i].number or 0))
        return [best]
    if min_c == 0 and max_c == 0:
        return []

    if max_c == 1:
        by_type: dict = defaultdict(list)
        for i, o in enumerate(opts):
            by_type[o.type].append(i)
        for ot in [OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                   OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT,
                   OptionType.CARD, OptionType.END]:
            if ot in by_type:
                return [random.choice(by_type[ot])]
        return [random.randint(0, n-1)]

    count = random.randint(max(1, min_c), max_c)
    return random.sample(range(n), min(count, n))


def _do_rollout(search_id, obs_after, your_index):
    cur_obs = obs_after
    cur_sid = search_id
    for _ in range(MAX_ROLLOUT_DEPTH):
        if cur_obs.current and cur_obs.current.result != -1:
            r = cur_obs.current.result
            return 1.0 if r == your_index else (0.0 if r == 2 else -1.0)
        choice = _simple_rollout_choice(cur_obs)
        try:
            nxt = search_step(cur_sid, choice)
        except Exception:
            return 0.0
        cur_obs = nxt.observation
        cur_sid = nxt.searchId
    return 0.0


# ---------------------------------------------------------------------------
# PIMC для MAIN select
# ---------------------------------------------------------------------------

def _should_use_pimc(obs: Observation) -> bool:
    """Используем PIMC только для MAIN select с реальным выбором.

    Критерий: есть хотя бы ATTACK и другие опции (нетривиальный выбор).
    """
    if not PIMC_ENABLED:
        return False
    sel = obs.select
    if sel is None or sel.context != SelectContext.MAIN:
        return False
    opts = sel.option
    n = len(opts)
    if n < 2:
        return False

    has_attack = any(o.type == OptionType.ATTACK for o in opts)
    has_non_attack = any(o.type != OptionType.ATTACK and o.type != OptionType.END for o in opts)

    # PIMC имеет смысл только когда есть настоящий выбор: атаковать vs что-то сделать
    return has_attack and has_non_attack


def _pimc_candidates(obs: Observation) -> list[list[int]]:
    """Кандидаты для PIMC: только разные типы действий из MAIN."""
    sel = obs.select
    opts = sel.option
    by_type: dict = defaultdict(list)
    for i, o in enumerate(opts):
        by_type[o.type].append(i)

    candidates = []
    # Атаки (каждая отдельно если их несколько)
    for i in by_type.get(OptionType.ATTACK, []):
        candidates.append([i])
    # По одному представителю других типов
    for ot in [OptionType.ABILITY, OptionType.EVOLVE, OptionType.PLAY,
               OptionType.ATTACH, OptionType.END]:
        if ot in by_type:
            candidates.append([random.choice(by_type[ot])])

    if not candidates:
        return [[0]]
    return candidates[:8]


def _pimc_evaluate(obs, candidates, time_budget):
    your_index = obs.current.yourIndex
    opp_idx = 1 - your_index
    n_cands = len(candidates)
    scores = [0.0] * n_cands
    counts = [0] * n_cands
    t_start = time.time()
    world = 0

    while time.time() - t_start < time_budget:
        try:
            your_deck, your_prize, opp_deck, opp_prize, opp_hand = \
                _build_search_inputs(obs, your_index)
            opp_act = _opp_active_ids(obs, opp_idx)
        except Exception:
            world += 1
            continue

        ci = world % n_cands
        action = candidates[ci]

        try:
            root = search_begin(
                obs,
                your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opp_deck, opponent_prize=opp_prize,
                opponent_hand=opp_hand, opponent_active=opp_act,
                manual_coin=False,
            )
        except Exception:
            world += 1
            continue

        try:
            try:
                after = search_step(root.searchId, action)
            except Exception:
                search_release(root.searchId)
                world += 1
                continue
            result = _do_rollout(after.searchId, after.observation, your_index)
            scores[ci] += result
            counts[ci] += 1
            _stats["rollouts_total"] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass

        world += 1
        if min(counts) >= N_WORLDS_MIN and time.time() - t_start > time_budget * 0.6:
            break

    avg = [(scores[i] / counts[i] if counts[i] > 0 else -999.0, i) for i in range(n_cands)]
    avg.sort(reverse=True)
    return candidates[avg[0][1]]


# ---------------------------------------------------------------------------
# Публичный агент
# ---------------------------------------------------------------------------

def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        _stats["decisions_trivial"] += 1
        return _deck_ids()

    n_options = len(sel.get("option") or [])
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _stats["decisions_trivial"] += 1
        return []

    try:
        obs = to_observation_class(obs_dict)

        # Проверяем стоит ли использовать PIMC
        if _should_use_pimc(obs):
            candidates = _pimc_candidates(obs)
            if len(candidates) > 1:
                result = _pimc_evaluate(obs, candidates, TIME_BUDGET_SEC)
                if (result
                        and all(0 <= i < n_options for i in result)
                        and len(result) >= min_count
                        and len(set(result)) == len(result)):
                    _stats["decisions_pimc"] += 1
                    return result

        # Для всех остальных: используем эвристику
        result = _heuristic_action(obs_dict)
        if (result is not None
                and all(0 <= i < n_options for i in result)
                and len(result) >= min_count
                and len(set(result)) == len(result)):
            _stats["decisions_heuristic"] += 1
            return result

    except Exception:
        pass

    _stats["decisions_trivial"] += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
