"""PIMC Guided — PIMC с guided rollout (эвристика v012 как rollout policy).

Ключевое улучшение над предыдущими версиями:
- Rollout использует _decide(obs) из v012 эвристики НАПРЯМУЮ (без конвертации в dict).
  Это устраняет технический блокер: внутри search_step возвращается Observation dataclass,
  а не dict. Теперь мы вызываем heuristic._decide(obs) с dataclass напрямую.
- Оба игрока в rollout управляются одной и той же эвристикой (зеркало).
- PIMC решает только ПЕРВЫЙ шаг (из кандидатов MAIN select).
- Для всех НЕ-MAIN решений используем эвристику напрямую (без PIMC).

Решение технического блокера (конвертация Observation → obs_dict):
  Вместо конвертации Observation dataclass → dict → Observation (туда-обратно),
  мы импортируем внутреннюю функцию _decide(obs: Observation) из v012 main.py.
  _decide уже принимает Observation dataclass — никакой конвертации не нужно.
  Fallback: если v012 недоступен, используется dataclasses.asdict() для конвертации.

[УПРОЩЕНИЕ] Модель оппонента: mirror (предполагаем Iono деку у оппонента).
  Rationale: ~85% поля играет Crustle, но топ-1 (onechan1 μ1312) играет Iono.
  Mirror — разумное начальное предположение для прототипа.

Параметры:
  TIME_BUDGET_SEC=3.0  — бюджет на ход
  N_WORLDS_MIN=3       — минимум миров на кандидата до ранней остановки
  MAX_CANDIDATES=8     — максимум кандидатов
  MAX_ROLLOUT_DEPTH=200 — максимум шагов rollout
  PIMC_ENABLED=True    — можно отключить для бенчмарка чистой эвристики
"""

from __future__ import annotations

import random
import time
import itertools
import sys
import os
import dataclasses
from collections import defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Предпочитаем cg_native (кроссплатформенный автодетект бинарника —
    # работает и на macOS arm64, и в Docker linux x86_64). Фоллбэк на
    # старый cg/ (Linux-only) для обратной совместимости, если cg_native
    # почему-то недоступен.
    from cg_native.api import (
        to_observation_class, search_begin, search_step, search_end, search_release,
        Observation, SelectContext, OptionType, SelectType,
    )
except ImportError:
    from cg.api import (
        to_observation_class, search_begin, search_step, search_end, search_release,
        Observation, SelectContext, OptionType, SelectType,
    )

# ---------------------------------------------------------------------------
# Iono дека (60 карт) — та же что в v012
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
N_WORLDS_MIN: int = 3
MAX_CANDIDATES: int = 8
MAX_COMBO_SIZE: int = 3
MAX_ROLLOUT_DEPTH: int = 200
PIMC_ENABLED: bool = True


# ---------------------------------------------------------------------------
# Загрузка v012 эвристики
# ---------------------------------------------------------------------------
_v012_mod = None
_v012_decide = None   # _decide(obs: Observation) -> list[int]
_v012_agent = None    # agent(obs_dict: dict) -> list[int]


def _load_v012() -> bool:
    """Загружает внутреннюю функцию _decide из v012.

    Ключевое решение конвертации: вместо Observation→dict→Observation,
    мы напрямую вызываем _decide(obs: Observation). v012/_decide принимает
    Observation dataclass, что идеально для rollout внутри search state.
    """
    global _v012_mod, _v012_decide, _v012_agent

    if _v012_decide is not None:
        return True

    # Ищем v012_ref или v012_iono_heur
    for relpath in ("submissions/v012_ref/main.py", "submissions/v012_iono_heur/main.py"):
        v012_path = os.path.join(_REPO, relpath)
        if os.path.exists(v012_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("_v012_guided", v012_path)
                mod = importlib.util.module_from_spec(spec)
                mod.__file__ = v012_path
                v012_dir = os.path.dirname(v012_path)
                if v012_dir not in sys.path:
                    sys.path.insert(0, v012_dir)
                spec.loader.exec_module(mod)
                _v012_mod = mod
                # Получаем внутренние функции напрямую
                _v012_decide = getattr(mod, "_decide", None)
                _v012_agent = getattr(mod, "agent", None)
                print(f"[PIMC-Guided] Loaded v012 from {relpath} "
                      f"(_decide={'OK' if _v012_decide else 'MISSING'})", flush=True)
                return _v012_decide is not None
            except Exception as e:
                print(f"[PIMC-Guided] Failed to load v012 from {relpath}: {e}", flush=True)
                continue

    print("[PIMC-Guided] WARNING: v012 not found, using fallback heuristic", flush=True)
    return False


def _obs_to_dict(obs: Observation) -> dict:
    """Конвертирует Observation dataclass в dict (fallback метод).

    Используется только если _decide недоступна напрямую.
    dataclasses.asdict() рекурсивно конвертирует вложенные датаклассы,
    IntEnum значения остаются как int — именно это нужно v012 agent(obs_dict).
    """
    def _convert(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        elif isinstance(obj, list):
            return [_convert(item) for item in obj]
        elif hasattr(obj, 'value'):  # IntEnum
            return obj.value
        else:
            return obj

    return _convert(obs)


# ---------------------------------------------------------------------------
# Guided rollout — использует v012 эвристику для обоих игроков
# ---------------------------------------------------------------------------

def _guided_choice_from_obs(obs: Observation) -> list[int]:
    """Выбор действия используя v012 эвристику напрямую с Observation dataclass.

    Три уровня fallback:
    1. _decide(obs) — прямой вызов внутренней функции v012 (без конвертации)
    2. agent(obs_dict) — через конвертацию dataclass→dict
    3. _simple_fallback(obs) — простые правила без v012
    """
    _load_v012()

    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    if n == 0:
        return []

    # Быстрый путь для тривиальных случаев
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
    max_c = min(sel.maxCount, n)
    if min_c == 0 and max_c == 0:
        return []

    # Уровень 1: прямой вызов _decide(obs: Observation)
    if _v012_decide is not None:
        try:
            result = _v012_decide(obs)
            if result is not None and all(0 <= i < n for i in result):
                return result
        except Exception:
            pass

    # Уровень 2: через конвертацию dict → v012 agent
    if _v012_agent is not None:
        try:
            obs_dict = _obs_to_dict(obs)
            result = _v012_agent(obs_dict)
            if result is not None and all(0 <= i < n for i in result):
                return result
        except Exception:
            pass

    # Уровень 3: простой fallback
    return _simple_fallback(obs)


def _simple_fallback(obs: Observation) -> list[int]:
    """Простая эвристика без v012: ATTACK > ABILITY > EVOLVE > PLAY > ATTACH > END."""
    sel = obs.select
    opts = sel.option
    n = len(opts)
    min_c = sel.minCount
    max_c = min(sel.maxCount, n)

    if max_c == 1:
        by_type: dict = defaultdict(list)
        for i, o in enumerate(opts):
            by_type[o.type].append(i)
        for ot in [OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                   OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT,
                   OptionType.CARD, OptionType.END]:
            if ot in by_type:
                return [random.choice(by_type[ot])]
        return [random.randint(0, n - 1)]

    count = random.randint(max(1, min_c), max_c)
    return random.sample(range(n), min(count, n))


def _do_guided_rollout(search_id: int, obs_after: Observation, your_index: int) -> float:
    """Guided rollout: оба игрока используют v012 эвристику.

    Возвращает: +1.0 выигрыш, 0.0 ничья или timeout, -1.0 проигрыш.
    """
    cur_obs = obs_after
    cur_sid = search_id

    for _ in range(MAX_ROLLOUT_DEPTH):
        if cur_obs.current and cur_obs.current.result != -1:
            r = cur_obs.current.result
            if r == your_index:
                return 1.0
            elif r == 2:
                return 0.0
            else:
                return -1.0

        choice = _guided_choice_from_obs(cur_obs)

        try:
            nxt = search_step(cur_sid, choice)
        except Exception:
            return 0.0

        cur_obs = nxt.observation
        cur_sid = nxt.searchId

    # Rollout не завершился за MAX_ROLLOUT_DEPTH шагов
    return 0.0


# ---------------------------------------------------------------------------
# Скрытое состояние оппонента
# ---------------------------------------------------------------------------

def _sample_hidden(known_ids: list[int], full_deck: list[int], count: int) -> list[int]:
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
    for p in list(opp.active or []) + list(opp.bench or []):
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


def _opp_active_ids(obs: Observation, opp_idx: int) -> list[int]:
    opp = obs.current.players[opp_idx]
    if opp.active and opp.active[0] is None:
        return [265]  # Voltorb как заглушка для лица-вниз активного
    return []


# ---------------------------------------------------------------------------
# Выбор кандидатов для MAIN select
# ---------------------------------------------------------------------------

def _smart_candidates(obs: Observation) -> list[list[int]]:
    """Для MAIN: по одному представителю каждого типа действия.
    Для других контекстов: все варианты (до MAX_CANDIDATES).
    """
    sel = obs.select
    opts = sel.option
    n = len(opts)
    min_c = sel.minCount
    max_c = sel.maxCount

    if min_c == 1 and max_c == 1:
        if sel.context == SelectContext.MAIN:
            seen: set = set()
            candidates = []
            by_type: dict = defaultdict(list)
            for i, o in enumerate(opts):
                by_type[o.type].append(i)

            for ot in [OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                       OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT,
                       OptionType.END, OptionType.CARD]:
                if ot in by_type and ot not in seen:
                    candidates.append([random.choice(by_type[ot])])
                    seen.add(ot)
            for ot, idxs in by_type.items():
                if ot not in seen:
                    candidates.append([random.choice(idxs)])
                    seen.add(ot)
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
    if len(sel.option) <= 1:
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
    count = max(1, sel.minCount)
    return list(range(min(count, n)))


# ---------------------------------------------------------------------------
# PIMC оценка кандидатов
# ---------------------------------------------------------------------------

# Статистика
_stats = {
    "rollouts_total": 0,
    "decisions_pimc": 0,
    "decisions_heuristic": 0,
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


def _evaluate_candidates(obs: Observation, candidates: list[list[int]],
                          time_budget: float) -> tuple[int, int]:
    """Оцениваем кандидатов через PIMC с guided rollout.

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
        try:
            your_deck, your_prize, opp_deck, opp_prize, opp_hand = \
                _build_search_inputs(obs, your_index)
            opp_act = _opp_active_ids(obs, opp_idx)
        except Exception:
            world_total += 1
            continue

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

            result = _do_guided_rollout(after.searchId, after.observation, your_index)
            scores[cand_idx] += result
            counts[cand_idx] += 1
            _stats["rollouts_total"] += 1

        finally:
            # ВАЖНО: всегда освобождаем память search tree
            try:
                search_end()
            except Exception:
                pass

        world_total += 1

        # Ранний выход если набрали достаточно миров и прошло >50% времени
        if min(counts) >= N_WORLDS_MIN and time.time() - t_start > time_budget * 0.5:
            break

    avg_scores = [(scores[i] / counts[i] if counts[i] > 0 else -999.0, i)
                  for i in range(n_cands)]
    # H3-4 bug-audit fix: sort по score ТОЛЬКО (stable). При равных score
    # сохраняется исходный порядок кандидатов — т.е. первый кандидат
    # (ATTACK в порядке _smart_candidates) побеждает, а НЕ последний (END/CARD).
    # Баг: tuple (score, idx).sort(reverse=True) при равных score выбирал
    # БОЛЬШИЙ idx (END/CARD) вместо меньшего (ATTACK) — 43% решений, 49% с
    # max score=1.0 (near-certain win vs random) → PIMC выбирал END вместо ATTACK.
    avg_scores.sort(key=lambda x: x[0], reverse=True)
    return avg_scores[0][1], world_total


# ---------------------------------------------------------------------------
# Публичный агент
# ---------------------------------------------------------------------------

def _deck_ids() -> list[int]:
    return list(IONO_DECK)


def agent(obs_dict: dict) -> list[int]:
    """PIMC Guided агент для Iono Electric деки.

    Первый вызов (obs.select is None): возвращаем деку.
    Не-MAIN контексты: v012 эвристика напрямую (без PIMC).
    MAIN контекст с реальным выбором: PIMC с guided rollout.
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

        # Тривиальные случаи — быстрый путь
        if _is_trivial(obs):
            _stats["decisions_trivial"] += 1
            return _trivial_choice(obs)

        # Проверяем нужен ли PIMC
        use_pimc = (
            PIMC_ENABLED
            and obs.select is not None
            and obs.select.context == SelectContext.MAIN
        )

        if use_pimc:
            candidates = _smart_candidates(obs)

            if len(candidates) == 1:
                # Только один кандидат — тривиально
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

        # Не-MAIN контексты или fallback: используем v012 эвристику
        _load_v012()
        if _v012_agent is not None:
            try:
                result = _v012_agent(obs_dict)
                if (result is not None
                        and all(0 <= i < n_options for i in result)
                        and len(result) >= min_count
                        and len(set(result)) == len(result)):
                    _stats["decisions_heuristic"] += 1
                    return result
            except Exception:
                pass

        # Последний fallback: guided choice напрямую
        result = _guided_choice_from_obs(obs)
        if (result is not None
                and all(0 <= i < n_options for i in result)
                and len(result) >= min_count
                and len(set(result)) == len(result)):
            _stats["decisions_heuristic"] += 1
            return result

    except Exception:
        pass

    # Trivial legal fallback
    _stats["decisions_trivial"] += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
