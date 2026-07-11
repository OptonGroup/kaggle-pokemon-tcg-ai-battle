"""PIMC (Perfect Information Monte Carlo) пилот для Iono Electric деки.

Версия 2: улучшения над прототипом:
- Умный выбор кандидатов: при MAIN select берём по одному представителю
  каждого типа действия (ATTACK, ABILITY, EVOLVE, PLAY-supporter, PLAY-item, ATTACH, END)
  вместо перебора всех опций.
- Эвристический rollout (heuristic_rollout): предпочитаем атаку/способности
  вместо чисто случайного выбора.
- Ограничение глубины rollout (MAX_ROLLOUT_DEPTH) для скорости.
- Бюджет по времени строго соблюдается.

Упрощения (помечены [УПРОЩЕНИЕ]):
- Модель оппонента: MIRROR (предполагаем Iono деку оппонента).
- Мульти-выбор при minCount>1: первые MAX_CANDIDATES комбинаций.
- Нет параллелизма (движок не thread-safe).
"""

from __future__ import annotations

import random
import time
import itertools
from collections import defaultdict

from cg.api import (
    to_observation_class, search_begin, search_step, search_end, search_release,
    Observation, SelectContext, OptionType, AreaType, EnergyType, SelectType,
)

# ---------------------------------------------------------------------------
# Iono дека (60 карт)
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

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
TIME_BUDGET_SEC: float = 3.0
N_WORLDS_MIN: int = 3          # минимум миров на кандидата
MAX_CANDIDATES: int = 8        # максимум кандидатов
MAX_COMBO_SIZE: int = 3
MAX_ROLLOUT_DEPTH: int = 150   # максимум шагов rollout (ограничивает длину партии)

# Приоритеты типов действий при выборе представителей для MAIN context
_ACTION_TYPE_PRIORITY = {
    OptionType.ATTACK:  10,
    OptionType.ABILITY: 9,
    OptionType.EVOLVE:  8,
    OptionType.PLAY:    7,
    OptionType.ATTACH:  6,
    OptionType.RETREAT: 5,
    OptionType.END:     1,
}

# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------
_stats = {
    "rollouts_total": 0,
    "decisions_pimc": 0,
    "decisions_trivial": 0,
    "time_total": 0.0,
    "worlds_per_decision": [],
}


def get_stats() -> dict:
    d = dict(_stats)
    wpd = d.pop("worlds_per_decision")
    if wpd:
        d["avg_worlds_per_decision"] = sum(wpd) / len(wpd)
        d["min_worlds"] = min(wpd)
        d["max_worlds"] = max(wpd)
    return d


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _deck_ids() -> list[int]:
    return list(IONO_DECK)


def _sample_hidden(known_ids: list[int], full_deck: list[int], count: int) -> list[int]:
    """Сэмплируем count карт из full_deck, исключая известные."""
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


def _opp_known(obs: Observation, opp_idx: int) -> list[int]:
    known = []
    opp = obs.current.players[opp_idx]
    for c in (opp.discard or []):
        known.append(c.id)
    for p in (opp.active or []):
        if p is not None:
            known.append(p.id)
            for ec in (p.energyCards or []):
                known.append(ec.id)
    for p in (opp.bench or []):
        if p is not None:
            known.append(p.id)
            for ec in (p.energyCards or []):
                known.append(ec.id)
    for c in (opp.prize or []):
        if c is not None:
            known.append(c.id)
    return known


def _build_search_inputs(obs: Observation, your_index: int):
    my = obs.current.players[your_index]
    opp_idx = 1 - your_index
    opp = obs.current.players[opp_idx]

    # Наша дека
    ydc = my.deckCount
    your_deck = (list(IONO_DECK) + [4] * 10)[:max(ydc, 0)]
    your_deck = your_deck[:ydc]

    # Наши призы
    your_prize = [p.id if p is not None else 4 for p in (my.prize or [])]

    # Дека оппонента — [УПРОЩЕНИЕ] mirror
    opp_full = list(IONO_DECK)
    opp_kn = _opp_known(obs, opp_idx)

    odc = opp.deckCount
    opp_deck = _sample_hidden(opp_kn, opp_full, odc)

    # Призы оппонента
    opp_prize = []
    for p in (opp.prize or []):
        if p is not None:
            opp_prize.append(p.id)
        else:
            pool = [x for x in opp_full if x not in opp_prize]
            opp_prize.append(random.choice(pool) if pool else 4)

    # Рука оппонента
    ohc = opp.handCount
    opp_hand = _sample_hidden(opp_kn, opp_full, ohc)

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand


def _opp_active_ids(obs: Observation, opp_idx: int) -> list[int]:
    opp = obs.current.players[opp_idx]
    if opp.active and opp.active[0] is None:
        return [265]  # Voltorb заглушка
    return []


# ---------------------------------------------------------------------------
# Эвристический rollout
# ---------------------------------------------------------------------------

def _heuristic_choice(obs: Observation) -> list[int]:
    """Быстрый эвристический выбор для rollout.

    Приоритеты: ATTACK > ABILITY > EVOLVE > PLAY > ATTACH > END.
    Внутри типа — случайный выбор.
    """
    sel = obs.select
    if sel is None:
        return []
    opts = sel.option
    n = len(opts)
    if n == 0:
        return []

    min_c = sel.minCount
    max_c = min(sel.maxCount, n)

    # Для YES/NO — всегда YES
    if sel.type == SelectType.YES_NO:
        for i, o in enumerate(opts):
            if o.type == OptionType.YES:
                return [i]
        return [0]

    # Для COUNT — максимальное число
    if sel.type == SelectType.COUNT:
        best = max(range(n), key=lambda i: (opts[i].number or 0))
        return [best]

    if min_c == 0 and max_c == 0:
        return []

    # Для одиночного выбора: по приоритету типа
    if max_c == 1:
        by_type: dict[OptionType, list[int]] = defaultdict(list)
        for i, o in enumerate(opts):
            by_type[o.type].append(i)

        for ot in [OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                   OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT,
                   OptionType.CARD, OptionType.END]:
            if ot in by_type:
                return [random.choice(by_type[ot])]

        return [random.randint(0, n-1)]

    # Мульти-выбор — случайный
    count = random.randint(max(1, min_c), max_c)
    return random.sample(range(n), min(count, n))


def _do_rollout(search_id: int, obs_after: Observation, your_index: int) -> float:
    """Эвристический rollout от текущего состояния.

    Возвращает: +1 выиграли, 0 ничья, -1 проиграли.
    """
    cur_obs = obs_after
    cur_sid = search_id

    for _ in range(MAX_ROLLOUT_DEPTH):
        if cur_obs.current and cur_obs.current.result != -1:
            result = cur_obs.current.result
            if result == your_index:
                return 1.0
            elif result == 2:
                return 0.0
            else:
                return -1.0

        choice = _heuristic_choice(cur_obs)

        try:
            nxt = search_step(cur_sid, choice)
        except Exception:
            return 0.0

        cur_obs = nxt.observation
        cur_sid = nxt.searchId

    # Rollout не завершён — нейтральный результат
    return 0.0


# ---------------------------------------------------------------------------
# Умный выбор кандидатов
# ---------------------------------------------------------------------------

def _smart_candidates(obs: Observation) -> list[list[int]]:
    """Для MAIN context: по одному представителю каждого типа действия.

    Гарантируем что перебираем не более MAX_CANDIDATES кандидатов.
    Для других контекстов: все индексы (или первые MAX_CANDIDATES).
    """
    sel = obs.select
    opts = sel.option
    n = len(opts)
    min_c = sel.minCount
    max_c = sel.maxCount

    # Одиночный выбор
    if min_c == 1 and max_c == 1:
        if sel.context == SelectContext.MAIN:
            # Группируем по типу, берём 1 представителя каждого типа
            seen_types: set = set()
            candidates = []
            # Сначала добавляем по приоритету типа
            type_order = [
                OptionType.ATTACK,
                OptionType.ABILITY,
                OptionType.EVOLVE,
                OptionType.PLAY,
                OptionType.ATTACH,
                OptionType.RETREAT,
                OptionType.END,
                OptionType.CARD,
            ]
            # Индексируем по типу
            by_type: dict = defaultdict(list)
            for i, o in enumerate(opts):
                by_type[o.type].append(i)

            for ot in type_order:
                if ot in by_type and ot not in seen_types:
                    candidates.append([random.choice(by_type[ot])])
                    seen_types.add(ot)

            # Остальные типы
            for ot, idxs in by_type.items():
                if ot not in seen_types:
                    candidates.append([random.choice(idxs)])
                    seen_types.add(ot)

            if not candidates:
                candidates = [[i] for i in range(min(n, MAX_CANDIDATES))]

            return candidates[:MAX_CANDIDATES]
        else:
            return [[i] for i in range(min(n, MAX_CANDIDATES))]

    # Мульти-выбор
    combo_size = min(min_c, MAX_COMBO_SIZE, n)
    candidates = []
    for combo in itertools.combinations(range(n), combo_size):
        candidates.append(list(combo))
        if len(candidates) >= MAX_CANDIDATES:
            break

    if not candidates:
        count = max(1, min_c)
        candidates = [list(range(min(count, n)))]

    return candidates


# ---------------------------------------------------------------------------
# Тривиальные решения
# ---------------------------------------------------------------------------

def _is_trivial(obs: Observation) -> bool:
    sel = obs.select
    if sel is None:
        return True
    n = len(sel.option)
    if n <= 1:
        return True
    if sel.type in (SelectType.YES_NO, SelectType.COUNT):
        return True
    return False


def _trivial_choice(obs: Observation) -> list[int]:
    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if sel.type == SelectType.YES_NO:
        for i, o in enumerate(sel.option):
            if o.type == OptionType.YES:
                return [i]
        return [0]
    if sel.type == SelectType.COUNT:
        best = max(range(n), key=lambda i: (sel.option[i].number or 0))
        return [best]
    min_c = sel.minCount
    count = max(1, min_c)
    return list(range(min(count, n)))


# ---------------------------------------------------------------------------
# PIMC оценка кандидатов
# ---------------------------------------------------------------------------

def _evaluate_candidates(obs: Observation, candidates: list[list[int]],
                          time_budget: float) -> tuple[int, int]:
    """Оцениваем кандидатов через PIMC.

    Возвращает (индекс лучшего кандидата, общее число миров).
    """
    your_index = obs.current.yourIndex
    opp_idx = 1 - your_index
    n_cands = len(candidates)

    if n_cands == 1:
        return 0, 0

    scores = [0.0] * n_cands
    counts = [0] * n_cands

    t_start = time.time()
    world_total = 0

    while time.time() - t_start < time_budget:
        # Сэмплируем мир
        try:
            your_deck, your_prize, opp_deck, opp_prize, opp_hand = \
                _build_search_inputs(obs, your_index)
            opp_act = _opp_active_ids(obs, opp_idx)
        except Exception:
            world_total += 1
            continue

        # Round-robin по кандидатам
        cand_idx = world_total % n_cands
        action = candidates[cand_idx]

        try:
            root = search_begin(
                obs,
                your_deck=your_deck,
                your_prize=your_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_act,
                manual_coin=False,
            )
        except Exception:
            world_total += 1
            continue

        try:
            try:
                after = search_step(root.searchId, action)
            except Exception:
                search_release(root.searchId)
                world_total += 1
                continue

            result = _do_rollout(after.searchId, after.observation, your_index)
            scores[cand_idx] += result
            counts[cand_idx] += 1
            _stats["rollouts_total"] += 1

        finally:
            try:
                search_end()
            except Exception:
                pass

        world_total += 1

        # Ранний выход если набрали достаточно
        if min(counts) >= N_WORLDS_MIN:
            remaining = time_budget - (time.time() - t_start)
            # Продолжаем пока есть время, но не дольше бюджета
            if remaining < 0.05:
                break

    avg_scores = [(scores[i] / counts[i] if counts[i] > 0 else -999.0, i)
                  for i in range(n_cands)]
    avg_scores.sort(reverse=True)
    return avg_scores[0][1], world_total


# ---------------------------------------------------------------------------
# Публичный агент
# ---------------------------------------------------------------------------

def agent(obs_dict: dict) -> list[int]:
    """PIMC агент для Iono Electric деки.

    Первый вызов (obs.select is None): возвращаем деку.
    Остальные: PIMC с бюджетом TIME_BUDGET_SEC секунд на ход.
    """
    t0 = time.time()

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

        if _is_trivial(obs):
            result = _trivial_choice(obs)
            _stats["decisions_trivial"] += 1
            return result

        candidates = _smart_candidates(obs)

        if len(candidates) == 1:
            _stats["decisions_trivial"] += 1
            return candidates[0]

        best_idx, n_worlds = _evaluate_candidates(obs, candidates, TIME_BUDGET_SEC)
        result = candidates[best_idx]

        _stats["decisions_pimc"] += 1
        _stats["time_total"] += time.time() - t0
        _stats["worlds_per_decision"].append(n_worlds)

        if (result
                and all(0 <= i < n_options for i in result)
                and len(result) >= min_count
                and len(set(result)) == len(result)):
            return result

    except Exception:
        pass

    # Fallback
    _stats["decisions_trivial"] += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
