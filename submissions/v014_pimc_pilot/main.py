"""v014_pimc_pilot — PIMC агент для Iono Electric деки. v2.

Улучшения: умные кандидаты (по типу действия), эвристический rollout,
ограничение глубины rollout, модель оппонента mirror.
"""
from __future__ import annotations

import os
import sys
import random
import time
import itertools
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/kaggle/working"
sys.path.insert(0, _HERE)

from cg.api import (
    to_observation_class, search_begin, search_step, search_end, search_release,
    Observation, SelectContext, OptionType, AreaType, EnergyType, SelectType,
)

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

TIME_BUDGET_SEC: float = 3.0
N_WORLDS_MIN: int = 3
MAX_CANDIDATES: int = 8
MAX_COMBO_SIZE: int = 3
MAX_ROLLOUT_DEPTH: int = 150


def _deck_ids():
    return list(IONO_DECK)


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


def _heuristic_choice(obs):
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
        by_type = defaultdict(list)
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
            if r == your_index:
                return 1.0
            elif r == 2:
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
    return 0.0


def _smart_candidates(obs):
    sel = obs.select
    opts = sel.option
    n = len(opts)
    min_c = sel.minCount
    max_c = sel.maxCount
    if min_c == 1 and max_c == 1:
        if sel.context == SelectContext.MAIN:
            seen = set()
            candidates = []
            by_type = defaultdict(list)
            for i, o in enumerate(opts):
                by_type[o.type].append(i)
            for ot in [OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                       OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT, OptionType.END]:
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


def _is_trivial(obs):
    sel = obs.select
    if sel is None:
        return True
    if len(sel.option) <= 1:
        return True
    if sel.type in (SelectType.YES_NO, SelectType.COUNT):
        return True
    return False


def _trivial_choice(obs):
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


def _evaluate_candidates(obs, candidates, time_budget):
    your_index = obs.current.yourIndex
    opp_idx = 1 - your_index
    n_cands = len(candidates)
    if n_cands == 1:
        return 0

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
            result = _do_rollout(after.searchId, after.observation, your_index)
            scores[cand_idx] += result
            counts[cand_idx] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass

        world_total += 1
        if min(counts) >= N_WORLDS_MIN:
            if time.time() - t_start > time_budget * 0.5:
                break

    avg = [(scores[i] / counts[i] if counts[i] > 0 else -999.0, i) for i in range(n_cands)]
    avg.sort(reverse=True)
    return avg[0][1]


def agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return _deck_ids()

    n_options = len(sel.get("option") or [])
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        return []

    try:
        obs = to_observation_class(obs_dict)
        if _is_trivial(obs):
            return _trivial_choice(obs)
        candidates = _smart_candidates(obs)
        if len(candidates) == 1:
            return candidates[0]
        best_idx = _evaluate_candidates(obs, candidates, TIME_BUDGET_SEC)
        result = candidates[best_idx]
        if (result
                and all(0 <= i < n_options for i in result)
                and len(result) >= min_count
                and len(set(result)) == len(result)):
            return result
    except Exception:
        pass

    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
