"""
v005_hybrid — Heuristic-IL Hybrid agent for Pokemon TCG AI Battle.

Architecture:
  - Lucario heuristic produces per-option scores (identical to v002 champion).
  - LightGBM IL ranker produces per-option scores.
  - Both score vectors are rank-normalized to [0,1].
  - combined[i] = (1-λ)*H_norm[i] + λ*IL_norm[i]
  - λ = HYBRID_LAMBDA env var (default 0.2). λ=0 → pure heuristic (sanity check).
  - Selection: top maxCount distinct indices from combined scores.
  - Never-throw: any exception → pure heuristic fallback → trivial fallback.

Instrumentation:
  - _n_heuristic_only: decisions where IL agreed with heuristic top-1
  - _n_il_override: decisions where IL changed the top-1 choice
  - _n_exceptions: exceptions caught (fell back to heuristic/trivial)
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Kaggle loads agents via exec() which does NOT set __file__, so we must not
# rely on it unconditionally.
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _DIR = _KAGGLE_AGENT_DIR
else:
    try:
        _DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
from collections import defaultdict

# ── Lambda parameter ──────────────────────────────────────────────────────────
# Read from env var HYBRID_LAMBDA at import time so the same main.py works for
# all λ values in the sweep; set λ=0 for a pure-heuristic sanity check.
try:
    _LAMBDA = float(os.environ.get("HYBRID_LAMBDA", "0.2"))
except (ValueError, TypeError):
    _LAMBDA = 0.2

# ── Deck loading ──────────────────────────────────────────────────────────────

def _find_file(name: str) -> str:
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv = _f.read().split("\n")
my_deck = [int(_csv[i]) for i in range(60)]

# ── Load LightGBM IL ranker ───────────────────────────────────────────────────
import lightgbm as lgb

_model_path = _find_file("il_ranker.txt")
_ranker: lgb.Booster = lgb.Booster(model_file=_model_path)

# ── Feature extraction (from features.py) ────────────────────────────────────
from features import build_feature_matrix  # noqa: E402

# ── CG API imports ────────────────────────────────────────────────────────────
from cg.api import (  # noqa: E402
    AreaType, CardType, EnergyType, Observation,
    SelectContext, OptionType, Card, Pokemon,
    all_card_data, to_observation_class,
)

# ── Card metadata ─────────────────────────────────────────────────────────────
_all_card = all_card_data()
_card_table = {c.cardId: c for c in _all_card}

# ── Lucario deck card IDs ─────────────────────────────────────────────────────
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

# ── Heuristic plan globals (reset per turn) ───────────────────────────────────

class _AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False


_plan = _AttackPlan()
_pre_turn = 0
_ability_used = False

# ── Instrumentation counters ──────────────────────────────────────────────────
# Named to match gate_runner.py's instrumentation block:
#   _n_model_decisions   = hybrid decisions where IL was active (λ>0) OR pure heuristic (λ=0)
#   _n_heuristic_fallback = decisions where IL overrode heuristic top-1
#   _n_trivial_fallback  = exception-caught fallbacks
_n_model_decisions: int = 0       # total non-exception decisions
_n_heuristic_fallback: int = 0    # IL changed heuristic's top-1 (only meaningful when λ>0)
_n_trivial_fallback: int = 0      # exceptions → fell back to heuristic/trivial


# ── Helper functions (heuristic) ──────────────────────────────────────────────

def _get_card_h(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def _prize_count_h(pokemon) -> int:
    data = _card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def _pokemon_score_h(pokemon) -> int:
    data = _card_table[pokemon.id]
    score = _prize_count_h(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    id_ = pokemon.id
    if id_ == 144 or id_ == 322 or id_ == 323 or id_ == 337:
        score -= 200
    if id_ == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


def _heuristic_scores(obs_dict: dict) -> tuple[list[int], list[float]]:
    """Compute heuristic per-option scores.

    Returns (desc_indices, raw_scores) where desc_indices is the sorted list
    (best first) and raw_scores[i] is the score for option i.
    """
    global _plan, _pre_turn, _ability_used

    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck, []

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

    if _pre_turn != state.turn:
        _pre_turn = state.turn
        _plan = _AttackPlan()
        _ability_used = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card is None:
            continue
        field_counts[card.id] += 1
        if card.id == Makuhita or card.id == Hariyama:
            if len(card.energies) >= 3:
                attacker2 = True
        elif card.id == Riolu or card.id == Mega_Lucario_ex:
            if len(card.energies) >= 2:
                attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = _get_card_h(obs, AreaType.HAND, o.index, my_index)
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = _get_card_h(obs, AreaType.HAND, o.index, my_index)
                if card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:
                    can_use_mega_brave = True

        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            energy_required = 1
                            base_damage = 130
                            base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            energy_required = 2
                            base_damage = 270
                        if my_prize == 2 or my_prize == 3:
                            base_score -= 500
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            energy_required = 1
                            base_damage = 70

                    if base_damage <= 0:
                        continue

                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            else:
                                more_energy = True
                        else:
                            continue

                    for j, op_pokemon in enumerate(op_cards):
                        if j != 0 and not can_op_switch:
                            break
                        damage = base_damage
                        data = _card_table[op_pokemon.id]
                        if data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif data.resistance == EnergyType.FIGHTING:
                            damage -= 30
                        prize = 0
                        score = _pokemon_score_h(op_pokemon)
                        if op_pokemon.hp <= damage:
                            prize = _prize_count_h(op_pokemon)
                        else:
                            score *= damage / op_pokemon.hp
                        score += base_score

                        if len(op_state.prize) <= prize:
                            score = 50000

                        if i == 0:
                            score += 220
                        if j == 0:
                            score += 300
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            _plan.attacker = i
                            _plan.target = j
                            _plan.attack_index = a
                            _plan.remain_hp = op_pokemon.hp - damage
                            _plan.energy = more_energy

    def energy_score_h(pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Makuhita or pokemon.id == Hariyama:
            if pokemon.id == Hariyama:
                score += 1
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    raw_scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = _get_card_h(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == _plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Lucario_ex:
                            if my_prize == 2 or my_prize == 3:
                                score += 8
                            else:
                                score += 20
                        elif card.id == Hariyama and energy_count >= 2:
                            score += 15
                        elif card.id == Makuhita and energy_count >= 2:
                            score += 10
                        elif card.id == Solrock:
                            score += 5
                        elif card.id == Riolu:
                            score += 4
                    else:
                        if o.index == _plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        if state.firstPlayer == my_index:
                            score = 2
                        else:
                            score = 4
                    elif card.id == Riolu:
                        score = 3
                    elif card.id == Makuhita:
                        score = 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        if field_counts[card.id] >= 1:
                            score -= 10
                        else:
                            score += 10
                    elif card.id == Hariyama:
                        if field_counts[Makuhita] >= 1:
                            score += 20
                        else:
                            score -= 20
                    elif card.id == Lunatone:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += 60
                    elif card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += 50
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score -= 150
                        elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                            score -= 3
                        else:
                            score += 40
                    elif card.id == Mega_Lucario_ex:
                        if field_counts[Riolu] >= 1:
                            score += 40
                        else:
                            score -= 15
                    elif card.id == Basic_Fighting_Energy:
                        if not _ability_used or not state.energyAttached:
                            score += 30
                        else:
                            score -= 1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score_h(card, o.area == AreaType.ACTIVE)
        elif o.type == OptionType.PLAY:
            card = _get_card_h(obs, AreaType.HAND, o.index, my_index)
            data = _card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Lunatone or card.id == Solrock:
                    if field_counts[card.id] >= 1:
                        score = -1
                elif card.id == Riolu:
                    if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                        score = -1
            else:
                score = 10000
                if card.id == Switch:
                    if _plan.attacker <= 0:
                        score = -1
                    else:
                        score = 6000
                elif card.id == Premium_Power_Pro:
                    if state.supporterPlayed and _plan.remain_hp <= 0:
                        score = -1
                    elif not can_attack:
                        if not state.supporterPlayed and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                            score = 3050
                        else:
                            score = -1
                    else:
                        score = 5000
                elif card.id == Boss_Orders:
                    if _plan.target >= 1:
                        score = 3200
                    else:
                        score = -1
                elif card.id == Carmine:
                    score = 3000
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Gravity_Mountain:
                    if stadium_id == 0:
                        score = -1
        elif o.type == OptionType.ATTACH:
            card = _get_card_h(obs, AreaType.HAND, o.index, my_index)
            pokemon = _get_card_h(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score_h(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if _plan.attacker == 0 and _plan.energy:
                        score += 200
                else:
                    if _plan.attacker == 1 + o.inPlayIndex and _plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = _get_card_h(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and _plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = _get_card_h(obs, o.area, o.index, my_index)
            if card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            if _plan.attacker >= 1:
                score = 2000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if _plan.attack_index == 1:
                if o.attackId == 983:  # Mega Brave
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100

        raw_scores.append(score)

    # Update ability_used flag (same logic as original heuristic)
    desc_indices = [i for i, _ in sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)]
    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = _get_card_h(obs, o.area, o.index, my_index)
            if card.id == Lunatone:
                _ability_used = True

    return desc_indices, raw_scores


def _rank_normalize(scores: list[float]) -> list[float]:
    """Rank-normalize a list of scores to [0, 1].

    Highest score → 1.0, lowest → 0.0.
    Ties share the same rank value.
    With n=1 everything maps to 1.0 (safe).
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # argsort descending: rank 0 = best
    indexed = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ranks = [0.0] * n
    for rank, idx in enumerate(indexed):
        ranks[idx] = float(rank)  # 0 = best, n-1 = worst

    # normalize to [0, 1]: best → 1.0, worst → 0.0
    return [1.0 - r / (n - 1) for r in ranks]


def _hybrid_decide(obs_dict: dict) -> list[int]:
    """Core hybrid decision logic."""
    global _n_model_decisions, _n_heuristic_fallback

    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    n_opts = len(opts)
    if n_opts == 0:
        return []

    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    k = max(min_count, 1)
    k = min(k, max_count, n_opts)

    # ── Heuristic scores ──────────────────────────────────────────────────────
    desc_indices, h_scores = _heuristic_scores(obs_dict)
    h_top1 = desc_indices[0] if desc_indices else 0

    # ── If λ = 0: pure heuristic, skip IL entirely ────────────────────────────
    if _LAMBDA == 0.0:
        _n_model_decisions += 1  # counts all decisions
        return desc_indices[:k]

    # ── IL scores ─────────────────────────────────────────────────────────────
    X = build_feature_matrix(obs_dict)
    il_scores_raw = _ranker.predict(X).tolist()

    # ── Rank-normalize both ───────────────────────────────────────────────────
    h_norm = _rank_normalize(h_scores)
    il_norm = _rank_normalize(il_scores_raw)

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = [(1.0 - _LAMBDA) * h_norm[i] + _LAMBDA * il_norm[i] for i in range(n_opts)]

    # Sort by combined descending
    order = sorted(range(n_opts), key=lambda i: combined[i], reverse=True)
    combined_top1 = order[0]

    _n_model_decisions += 1
    # Track IL override (heuristic_fallback reused as "IL changed top-1" counter)
    if combined_top1 != h_top1:
        _n_heuristic_fallback += 1  # = IL_override count

    return order[:k]


# ── Public agent function ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point.

    Priority:
      1. Hybrid decision (heuristic-led + IL blend).
      2. Pure heuristic fallback (if hybrid throws).
      3. Trivial fallback: first minCount indices.
    """
    global _n_trivial_fallback

    # Deck selection step
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    # 1. Try hybrid
    try:
        result = _hybrid_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result):
            return result
    except Exception:
        pass

    # 2. Pure heuristic fallback
    _n_trivial_fallback += 1
    try:
        desc_indices, _ = _heuristic_scores(obs_dict)
        k = max(min_count, 1)
        k = min(k, max_count, n_options)
        result = desc_indices[:k]
        if result is not None and len(result) >= min_count:
            return result
    except Exception:
        pass

    # 3. Trivial fallback — always legal
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
