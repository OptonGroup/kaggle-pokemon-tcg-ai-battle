"""Iono Electric Swarm heuristic agent — hand-written rule-based pilot.

Deck composition (60 cards):
  - Iono's Voltorb (265) x3       Basic Lightning, 70HP, Voltaic Chain: 20+20×(L-energy on all Iono Pokemon)
  - Iono's Tadbulb (268) x3       Basic Lightning, 60HP, Tiny Charge: 30 (L+1 cost)
  - Iono's Bellibolt ex (269) x3  Stage1 ex Lightning, 280HP, Ability: Electric Streamer (attach L from hand
                                   to any Iono Pokemon unlimited times/turn), Thunderous Bolt: 230 (3L+1),
                                   but can't attack next turn after using it.
  - Iono's Wattrel (270) x3       Basic Lightning, 60HP, Quick Attack: 10+flip20 (L cost)
  - Iono's Kilowattrel (271) x3   Stage1 Lightning, 120HP, Ability: Flashing Draw (discard L → draw to 6),
                                   Mach Bolt: 70 (L+2 cost)
  - Canari (1233) x4              Supporter: discard 1 card → search deck for up to 4 L Pokemon
  - Lillie's Determination (1227) x4  Supporter: shuffle hand into deck → draw 6 (or 8 if 6 prizes left)
  - Buddy-Buddy Poffin (1086) x3  Item: search deck for up to 2 Basic Pokemon with 70HP or less → bench
  - Ultra Ball (1121) x3          Item: discard 2 cards → search deck for any Pokemon
  - Levincia (1254) x3            Stadium: once per player per turn: put up to 2 Basic L from discard to hand
  - Night Stretcher (1097) x2     Item: put Pokemon or Basic Energy from discard to hand
  - Poké Pad (1152) x2            Item: search deck for non-Rule-Box Pokemon → hand
  - Max Rod (1110) x1             Item (ACE SPEC): put up to 5 Pokemon/Basic Energy from discard to hand
  - Energy Retrieval (1118) x1    Item: put up to 2 Basic Energy from discard to hand
  - Basic Lightning Energy (4) x22

Strategy summary:
  Core loop: fill bench with Iono's basics (Voltorb, Wattrel, Tadbulb), evolve to Bellibolt ex /
  Kilowattrel. Bellibolt ex's Electric Streamer ability floods energy onto Voltorbs, supercharging
  Voltaic Chain (20 + 20×N lightning on all Iono Pokemon). A full bench with energy everywhere makes
  Voltorb hit for 100-160+. Kilowattrel's Flashing Draw ability (discard 1 L → draw to 6) fuels hand.

  Priority hierarchy:
    1. Bellibolt ex ability → attach as many L energies as possible
    2. Kilowattrel ability → draw when hand is thin (≤4 cards) and we have a L on it
    3. Spread energy to build board (all attackers up to their threshold)
    4. Attack: prefer lethal; Voltorb Voltaic Chain scales with board L-energy
    5. Play supporter before attacking (Canari > Lillie's Determination for setup)
    6. Bench basics aggressively (Buddy-Buddy Poffin, Canari)
    7. Evolve when ready
    8. Retreat from damaged / unenergized active to a charged attacker
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Optional

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, to_observation_class,
)

# ---------------------------------------------------------------------------
# Deck constants
# ---------------------------------------------------------------------------
Iono_Voltorb     = 265
Iono_Tadbulb     = 268
Iono_Bellibolt   = 269  # Bellibolt ex (stage1 of Tadbulb)
Iono_Wattrel     = 270
Iono_Kilowattrel = 271  # stage1 of Wattrel

Canari            = 1233
Lillie_Det        = 1227
Buddy_Buddy_Poffin = 1086
Ultra_Ball        = 1121
Levincia          = 1254
Night_Stretcher   = 1097
Poke_Pad          = 1152
Max_Rod           = 1110
Energy_Retrieval  = 1118
Basic_L_Energy    = 4

IONO_BASICS   = {Iono_Voltorb, Iono_Tadbulb, Iono_Wattrel}
IONO_POKEMON  = {Iono_Voltorb, Iono_Tadbulb, Iono_Bellibolt, Iono_Wattrel, Iono_Kilowattrel}
IONO_EVOLUTIONS = {Iono_Bellibolt, Iono_Kilowattrel}

# Crustle (Mysterious Rock Inn makes ex deal 0 damage) — only against Crustle do we avoid ex attacks
CRUSTLE_ID = 345

# Fighting-type Pokemon IDs (known dangerous Fighting attackers that exploit our L-weakness)
# Our Lightning Pokemon are all weak to Fighting, so 70HP Voltorbs die instantly.
FIGHTING_POKEMON_IDS = {
    673, 674,   # Makuhita, Hariyama
    677, 678,   # Riolu, Mega Lucario ex
    679, 680,   # other fighting lines
}

# Energy thresholds per attacker (min to attack / ideal)
ENERGY_MIN   = {Iono_Voltorb: 2, Iono_Tadbulb: 2, Iono_Wattrel: 1,
                Iono_Bellibolt: 4, Iono_Kilowattrel: 3}
ENERGY_IDEAL = {Iono_Voltorb: 3, Iono_Tadbulb: 2, Iono_Wattrel: 2,
                Iono_Bellibolt: 4, Iono_Kilowattrel: 3}


# ---------------------------------------------------------------------------
# Global card database
# ---------------------------------------------------------------------------
_all_card = all_card_data()
card_table = {c.cardId: c for c in _all_card}


# ---------------------------------------------------------------------------
# Instrumentation counters (gate_runner compatible)
# ---------------------------------------------------------------------------
_n_model_decisions: int = 0   # heuristic decisions
_n_heuristic_fallback: int = 0  # kept for compat (always 0)
_n_trivial_fallback: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Optional[Pokemon | Card]:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:   return obs.select.deck[index]
        case AreaType.HAND:   return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE:  return ps.active[index]
        case AreaType.BENCH:   return ps.bench[index]
        case AreaType.PRIZE:   return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _:                return None


def _prize_count(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    # megaEx gives 3 prizes (Mega Evolution Pokémon ex); ex gives 2; basics give 1
    if getattr(data, 'megaEx', False):
        count = 3
    elif data.ex:
        count = 2
    else:
        count = 1
    return max(0, count)


def _total_iono_lightning(my_state) -> int:
    """Count total Lightning energies attached to all Iono's Pokemon on board."""
    total = 0
    for poke in (my_state.active or []) + (my_state.bench or []):
        if poke is None:
            continue
        if poke.id in IONO_POKEMON:
            total += len(poke.energies)
    return total


def _voltaic_chain_damage(total_l: int) -> int:
    """Voltorb's Voltaic Chain: base 20 + 20 per Lightning energy on all Iono Pokemon."""
    return 20 + 20 * total_l


def _effective_damage(base_dmg: int, target: Pokemon) -> int:
    data = card_table.get(target.id)
    if data is None:
        return base_dmg
    dmg = base_dmg
    if data.weakness == EnergyType.LIGHTNING:
        dmg *= 2
    elif data.weakness == EnergyType.FIGHTING:
        pass  # our lightning has no fighting weakness overlap
    if data.resistance == EnergyType.LIGHTNING:
        dmg -= 30
    return max(0, dmg)


def _opp_has_fighting(opp_field: list) -> bool:
    """Return True if the opponent appears to be running a Fighting-type deck.

    We check by known Fighting Pokemon IDs and by energies attached
    (Pokemon.energies is a list[EnergyType] integers).
    """
    for poke in opp_field:
        if poke is None:
            continue
        if poke.id in FIGHTING_POKEMON_IDS:
            return True
        # Pokemon.energies is list[EnergyType] (IntEnum) — check directly
        for e in (poke.energies or []):
            try:
                if int(e) == int(EnergyType.FIGHTING):
                    return True
            except Exception:
                pass
    return False


def _opp_has_crustle(opp_field: list) -> bool:
    """Return True if the opponent has Crustle on their board (ex attacks deal 0 to it)."""
    return any(poke is not None and poke.id == CRUSTLE_ID for poke in opp_field)


def _pokemon_value(pokemon: Pokemon) -> int:
    """Target priority score for opponent Pokemon."""
    data = card_table.get(pokemon.id)
    if data is None:
        return 100
    score = _prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    score += pokemon.hp
    return score


def _attacker_energy_score(pokemon: Pokemon, already_has_attacker: bool, is_active: bool,
                           opp_fighting: bool = False, opp_crustle: bool = False) -> int:
    """Score for attaching energy to this Pokemon (higher = better target)."""
    pid = pokemon.id
    e = len(pokemon.energies)
    ideal = ENERGY_IDEAL.get(pid, 2)
    score = 5000
    if is_active:
        score += 20

    if pid == Iono_Voltorb:
        # Voltorb scales with total energy; always want more, but diminishing returns once many attached
        if e < ideal:
            score += 200
        elif e < ideal + 2:
            score += 50
        else:
            score -= 100
        # Against Fighting: slight preference away from Voltorb but still charge it
        # as a secondary attacker (needed when Bellibolt is on Thunderous Bolt cooldown).
        if opp_fighting:
            score -= 150
    elif pid == Iono_Bellibolt:
        # Don't energy-attach to Bellibolt directly — use ability instead; but ability bypass handled elsewhere
        if e < 4:
            score += 200  # Slightly higher base to compete with Voltorb
        elif e == 4:
            score -= 300  # At ideal — deprioritize (let others get energy)
        else:
            score -= 500  # Over-charged — strongly avoid
        # Against Fighting: Bellibolt ex is the primary — prioritise charging it
        # But only until it has enough (4 energy), then let other attackers charge
        if opp_fighting and not opp_crustle:
            if e < 4:
                score += 500  # Large bonus: get Bellibolt charged ASAP
            # Once at 4 energy, no extra bonus — let Voltorb/Kilowattrel charge too
    elif pid == Iono_Kilowattrel:
        if e < 3:
            score += 150
        elif e == 3:
            score += 50
        else:
            score -= 100
    elif pid == Iono_Tadbulb:
        if e < 2:
            score += 120
        else:
            score -= 100
        # Against Fighting: tadbulb will evolve into Bellibolt — still useful to charge
        if opp_fighting:
            score += 50
    elif pid == Iono_Wattrel:
        if e < 1:
            score += 100
        else:
            score -= 50
        # Against Fighting: very fragile (60HP) — slight deprioritization
        if opp_fighting:
            score -= 100
    else:
        score -= 500  # Don't attach to non-Iono Pokemon

    if already_has_attacker:
        # We have a charged attacker — deprioritize over-charging a second one
        if e >= ideal:
            score -= 200

    return score


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

class _TurnPlan:
    attacker_idx: int = -1   # 0=active, 1+=bench index+1
    target_idx: int = 0      # 0=active opponent
    attack_idx: int = 0
    needs_energy: bool = False
    lethal_dmg: int = 0


_plan = _TurnPlan()
_prev_turn = -1
_bellibolt_ability_used = False  # ability use tracking for bellibolt
_kilowattrel_ability_used = False


def _decide(obs: Observation) -> list[int]:
    global _plan, _prev_turn, _bellibolt_ability_used, _kilowattrel_ability_used

    if obs.select is None:
        raise RuntimeError("No select — should not reach here")

    state = obs.current
    select = obs.select
    ctx = select.context
    my_idx = state.yourIndex
    my = state.players[my_idx]
    opp = state.players[1 - my_idx]

    # Reset per-turn state
    if _prev_turn != state.turn:
        _prev_turn = state.turn
        _plan = _TurnPlan()
        _bellibolt_ability_used = False
        _kilowattrel_ability_used = False

    # --- Build inventory counts ---
    hand_cnt: dict[int, int] = defaultdict(int)
    field_cnt: dict[int, int] = defaultdict(int)
    discard_cnt: dict[int, int] = defaultdict(int)

    for c in my.hand:
        hand_cnt[c.id] += 1
    for c in (my.active or []) + (my.bench or []):
        if c:
            field_cnt[c.id] += 1
    for c in my.discard:
        discard_cnt[c.id] += 1

    # Opponent inventory (active may be empty list during setup phases)
    opp_active_list = [opp.active[0]] if opp.active else []
    opp_field: list[Pokemon] = opp_active_list + [p for p in opp.bench if p is not None]

    # My field (active=index0, bench starts at 1) — active may be empty during setup
    my_active_list = [my.active[0]] if my.active else []
    my_field: list[Pokemon] = my_active_list + [p for p in my.bench if p is not None]

    # Total lightning on board
    total_l = _total_iono_lightning(my)

    # Stadium
    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    # Opponent-awareness flags (computed once per decision step)
    try:
        opp_fighting = _opp_has_fighting(opp_field)
        opp_crustle  = _opp_has_crustle(opp_field)
    except Exception:
        opp_fighting = False
        opp_crustle  = False

    # --- Pre-compute attack plan on MAIN context ---
    if ctx == SelectContext.MAIN and state.turn >= 2:
        _compute_attack_plan(obs, my, opp, my_field, opp_field, total_l, my_idx)

    # --- Score each option ---
    scores: list[int] = []
    for o in select.option:
        s = _score_option(obs, o, ctx, my, opp, my_field, opp_field,
                          hand_cnt, field_cnt, discard_cnt,
                          total_l, stadium_id, my_idx, state,
                          opp_fighting=opp_fighting, opp_crustle=opp_crustle)
        scores.append(s)

    desc = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Update ability_used tracking
    if ctx == SelectContext.MAIN:
        top_opt = select.option[desc[0]]
        if top_opt.type == OptionType.ABILITY:
            c = _get_card(obs, top_opt.area, top_opt.index, my_idx)
            if c and c.id == Iono_Bellibolt:
                _bellibolt_ability_used = True
            if c and c.id == Iono_Kilowattrel:
                _kilowattrel_ability_used = True

    return desc[:select.maxCount]


def _compute_attack_plan(obs, my, opp, my_field, opp_field, total_l, my_idx):
    """Compute which attacker/target/attack gives the best outcome this turn."""
    global _plan
    state = obs.current
    select = obs.select

    # Can we switch to bench attacker?
    can_switch = any(
        o.type in (OptionType.RETREAT,) or
        (o.type == OptionType.PLAY and _hand_card_id(obs, o, my_idx) in (1123,))  # Switch item
        for o in select.option
    )
    can_boss = any(
        o.type == OptionType.PLAY and _hand_card_id(obs, o, my_idx) in (1182, 1192)
        for o in select.option
    )

    # Opponent-awareness flags
    opp_fighting = _opp_has_fighting(opp_field)
    opp_crustle  = _opp_has_crustle(opp_field)

    best_score = -999999

    for i, my_poke in enumerate(my_field):
        if i > 0 and not can_switch:
            continue
        if my_poke is None or my_poke.id not in IONO_POKEMON:
            continue

        for a_idx in range(2):
            base_dmg, e_req, can_use_it = _attack_params(my_poke, a_idx, total_l, state)
            if not can_use_it:
                continue

            # Against Crustle: ex attacks deal 0 — skip Bellibolt ex
            if opp_crustle and my_poke.id == Iono_Bellibolt:
                continue

            # Check energy availability
            e_have = len(my_poke.energies)
            needs_e = False
            if e_have < e_req:
                # Can we attach one more?
                if hand_has_energy(obs, my_idx) and not state.energyAttached:
                    e_have += 1
                    if e_have < e_req:
                        continue
                    needs_e = True
                else:
                    continue

            for j, op in enumerate(opp_field):
                if j > 0 and not can_boss:
                    break
                eff_dmg = _effective_damage(base_dmg, op)
                val = _pokemon_value(op)

                if op.hp <= eff_dmg:
                    # Lethal
                    score = 50000 + val + _prize_count(op) * 20000
                    if len(opp.prize) - _prize_count(op) <= 0:
                        score += 200000  # game over!
                else:
                    score = val * eff_dmg // max(1, op.hp)

                if i == 0:
                    score += 500  # prefer active attacker (no switch cost)
                if j == 0:
                    score += 200  # prefer hitting active (no boss needed)
                score += e_have * 10

                # --- Opponent-aware attacker adjustments ---
                if opp_fighting:
                    # Our Lightning Pokemon are weak to Fighting (2× incoming damage).
                    # Bellibolt ex (280HP) is our bulkiest attacker; prefer it for big targets.
                    # However, allow Voltorb to hit low-HP targets (Riolu 80HP) — don't over-penalize.
                    if my_poke.id == Iono_Bellibolt:
                        score += 15000  # 280HP — primary vs Fighting; big bonus to beat basics
                    elif my_poke.id == Iono_Kilowattrel:
                        score += 2000   # 120HP — secondary attacker, OK vs low-HP targets
                    elif my_poke.id == Iono_Voltorb:
                        # Allow Voltorb if it's lethal (1-shot target) or Bellibolt unavailable
                        bellibolt_available = any(
                            p is not None and p.id == Iono_Bellibolt and len(p.energies) >= 4
                            for p in my_field
                        )
                        if not bellibolt_available:
                            pass  # No penalty if Bellibolt not ready — Voltorb is our only option
                        elif op.hp <= eff_dmg:
                            pass  # Lethal hit — always OK
                        else:
                            score -= 10000  # Non-lethal with Bellibolt available — don't use Voltorb
                    elif my_poke.id in (Iono_Tadbulb, Iono_Wattrel):
                        score -= 8000   # 60HP basics — very fragile, avoid as primary

                if score > best_score:
                    best_score = score
                    _plan.attacker_idx = i
                    _plan.target_idx = j
                    _plan.attack_idx = a_idx
                    _plan.needs_energy = needs_e
                    _plan.lethal_dmg = eff_dmg - op.hp if op.hp <= eff_dmg else -(op.hp - eff_dmg)


def _attack_params(poke: Pokemon, a_idx: int, total_l: int, state) -> tuple[int, int, bool]:
    """Return (base_damage, energy_required, can_use).
    For Bellibolt ex we track the 'can't attack next turn' status via energies pattern.
    """
    pid = poke.id
    if pid == Iono_Voltorb:
        if a_idx == 0:
            return _voltaic_chain_damage(total_l), 2, True
        return 0, 0, False
    elif pid == Iono_Tadbulb:
        if a_idx == 0:
            return 30, 2, True
        return 0, 0, False
    elif pid == Iono_Bellibolt:
        if a_idx == 0:
            return 230, 4, True  # Thunderous Bolt
        return 0, 0, False
    elif pid == Iono_Wattrel:
        if a_idx == 0:
            return 10, 1, True  # Quick Attack (ignore flip; treat as 10+10=20 expected)
        return 0, 0, False
    elif pid == Iono_Kilowattrel:
        if a_idx == 0:
            return 70, 3, True  # Mach Bolt
        return 0, 0, False
    return 0, 0, False


def hand_has_energy(obs: Observation, my_idx: int) -> bool:
    return any(c.id == Basic_L_Energy for c in obs.current.players[my_idx].hand)


def _hand_card_id(obs: Observation, o, my_idx: int) -> int:
    c = _get_card(obs, AreaType.HAND, o.index, my_idx)
    return c.id if c else -1


def _score_option(obs, o, ctx, my, opp, my_field, opp_field,
                  hand_cnt, field_cnt, discard_cnt,
                  total_l, stadium_id, my_idx, state,
                  opp_fighting: bool = False, opp_crustle: bool = False) -> int:
    score = 0

    if o.type == OptionType.NUMBER:
        return o.number

    if o.type == OptionType.YES:
        return 1

    # ---- CARD selections (deck-look, prize, switch targets, etc.) ----
    if o.type == OptionType.CARD:
        card = _get_card(obs, o.area, o.index, o.playerIndex)
        if card is None:
            return 0

        e_cnt = len(card.energies) if isinstance(card, Pokemon) else 0

        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            if o.playerIndex == my_idx:
                score += e_cnt * 3
                # Prefer the planned attacker
                if _plan.attacker_idx > 0 and o.index == _plan.attacker_idx - 1:
                    score += 500
                # Priority for good attackers — boosted by opponent-awareness
                if card.id == Iono_Bellibolt:
                    score += 30 + (100 if opp_fighting and not opp_crustle else 0)
                elif card.id == Iono_Kilowattrel:
                    score += 20
                elif card.id == Iono_Voltorb:
                    score += 15 - (50 if opp_fighting else 0)  # penalise vs Fighting
                elif card.id == Iono_Wattrel:
                    score += 5 - (30 if opp_fighting else 0)
            else:
                # Opponent switch target — pick based on attack plan
                if _plan.target_idx > 0 and o.index == _plan.target_idx - 1:
                    score += 500

        elif ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            # Turn 1 setup: normally prefer Voltorb (cheap, chains scale), then Wattrel.
            # Against Fighting: prefer Tadbulb (→Bellibolt ex, 280HP) to avoid fast KO.
            if opp_fighting:
                if card.id == Iono_Tadbulb:
                    score = 5
                elif card.id == Iono_Voltorb:
                    score = 3  # still okay if no Tadbulb, but not ideal
                elif card.id == Iono_Wattrel:
                    score = 2
                elif card.id == Iono_Bellibolt:
                    score = 1
                elif card.id == Iono_Kilowattrel:
                    score = 0
            else:
                if card.id == Iono_Voltorb:
                    score = 5
                elif card.id == Iono_Wattrel:
                    score = 4
                elif card.id == Iono_Tadbulb:
                    score = 3
                elif card.id == Iono_Bellibolt:
                    score = 2
                elif card.id == Iono_Kilowattrel:
                    score = 1

        elif ctx == SelectContext.TO_HAND:
            # Prize selection, deck search results, Looking area, etc.
            score = 200 - hand_cnt[card.id] * 50
            if card.id == Basic_L_Energy:
                score += 30
            elif card.id == Iono_Bellibolt:
                if field_cnt.get(Iono_Tadbulb, 0) >= 1:
                    score += 100
                else:
                    score += 60
            elif card.id == Iono_Tadbulb:
                if field_cnt.get(Iono_Bellibolt, 0) + field_cnt.get(Iono_Tadbulb, 0) >= 2:
                    score -= 50
                else:
                    score += 60
            elif card.id == Iono_Voltorb:
                if field_cnt.get(Iono_Voltorb, 0) >= 3:
                    score -= 80
                else:
                    score += 50
            elif card.id == Iono_Kilowattrel:
                if field_cnt.get(Iono_Wattrel, 0) >= 1:
                    score += 80
                elif field_cnt.get(Iono_Kilowattrel, 0) == 0:
                    score += 40
                else:
                    score -= 30
            elif card.id == Iono_Wattrel:
                if field_cnt.get(Iono_Kilowattrel, 0) + field_cnt.get(Iono_Wattrel, 0) >= 2:
                    score -= 50
                else:
                    score += 40
            elif card.id == Canari:
                score += 40
            elif card.id == Lillie_Det:
                score += 35
            elif card.id == Buddy_Buddy_Poffin:
                score += 25
            elif card.id == Ultra_Ball:
                score += 20
            elif card.id == Levincia:
                score += 15
            elif card.id == Night_Stretcher:
                score += 15
            elif card.id == Max_Rod:
                score += 20
            elif card.id == Energy_Retrieval:
                score += 10

        elif ctx == SelectContext.ATTACH_FROM:
            # Bellibolt ex Electric Streamer ability: pick best target for L energy
            if isinstance(card, Pokemon):
                already = any(
                    p is not None and p.id in IONO_POKEMON and len(p.energies) >= ENERGY_MIN.get(p.id, 2)
                    for p in (my.active or []) + (my.bench or [])
                )
                score = _attacker_energy_score(card, already, o.area == AreaType.ACTIVE,
                                               opp_fighting=opp_fighting, opp_crustle=opp_crustle)
            else:
                score = 0

        return score

    # ---- PLAY (item / supporter / stadium from hand) ----
    if o.type == OptionType.PLAY:
        card = _get_card(obs, AreaType.HAND, o.index, my_idx)
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0

        if data.cardType == CardType.POKEMON:
            # Bench a basic Pokemon
            score = 20000
            if card.id == Iono_Voltorb:
                if field_cnt.get(Iono_Voltorb, 0) >= 3:
                    score = -1  # No more than 3
                else:
                    score += 50
            elif card.id == Iono_Tadbulb:
                if field_cnt.get(Iono_Tadbulb, 0) + field_cnt.get(Iono_Bellibolt, 0) >= 3:
                    score = -1
                else:
                    score += 40
            elif card.id == Iono_Wattrel:
                if field_cnt.get(Iono_Wattrel, 0) + field_cnt.get(Iono_Kilowattrel, 0) >= 2:
                    score = -1
                else:
                    score += 30
        else:
            cid = card.id

            if cid == Canari:
                # Supporter: discard 1 card → search up to 4 L Pokemon
                # Great early, less useful when bench is full
                bench_pokemon = sum(1 for p in my.bench if p is not None)
                if state.supporterPlayed:
                    score = -1
                elif bench_pokemon < 4:
                    score = 8000  # High priority: fills bench
                else:
                    score = 4000  # Still useful for hand quality

            elif cid == Lillie_Det:
                # Supporter: shuffle hand into deck, draw 6 (or 8 if 6 prizes)
                if state.supporterPlayed:
                    score = -1
                elif len(my.hand) <= 3:
                    score = 9000  # Desperately need cards
                elif len(my.hand) <= 5:
                    score = 6000
                else:
                    score = 3000  # Still OK

            elif cid == Buddy_Buddy_Poffin:
                # Item: search up to 2 basics ≤70HP → bench
                bench_pokemon = sum(1 for p in my.bench if p is not None)
                if bench_pokemon < 4:
                    score = 7000
                else:
                    score = -1  # Bench full

            elif cid == Ultra_Ball:
                # Item: discard 2 → get any Pokemon
                # Use if hand is big enough (need 2 discards + itself = 3 cards)
                if len(my.hand) >= 3:
                    # Need a Pokemon we're missing
                    missing_bellibolt = (field_cnt.get(Iono_Bellibolt, 0) == 0 and
                                         field_cnt.get(Iono_Tadbulb, 0) >= 1)
                    missing_kilowattrel = (field_cnt.get(Iono_Kilowattrel, 0) == 0 and
                                           field_cnt.get(Iono_Wattrel, 0) >= 1)
                    if missing_bellibolt or missing_kilowattrel:
                        score = 6500  # Priority: get evolution targets
                    else:
                        score = 4500
                else:
                    score = -1

            elif cid == Poke_Pad:
                # Item: search non-Rule-Box Pokemon (no ex/V)
                bench_pokemon = sum(1 for p in my.bench if p is not None)
                if bench_pokemon < 4:
                    score = 6000
                else:
                    score = 3000

            elif cid == Levincia:
                # Stadium: put up to 2 Basic L from discard to hand (once per turn)
                if stadium_id == Levincia:
                    score = -1  # Already active
                elif discard_cnt.get(Basic_L_Energy, 0) >= 1:
                    score = 4000
                else:
                    score = 1000

            elif cid == Night_Stretcher:
                # Item: recover Pokemon or Basic Energy from discard
                has_poke_in_discard = any(
                    card_table.get(c.id) and card_table[c.id].cardType == CardType.POKEMON
                    for c in my.discard
                )
                has_energy_in_discard = discard_cnt.get(Basic_L_Energy, 0) > 0
                if has_poke_in_discard or has_energy_in_discard:
                    score = 5000
                else:
                    score = -1

            elif cid == Max_Rod:
                # ACE SPEC: recover up to 5 Pokemon/Energy from discard
                total_discard = sum(
                    1 for c in my.discard
                    if (card_table.get(c.id) and card_table[c.id].cardType == CardType.POKEMON)
                    or c.id == Basic_L_Energy
                )
                if total_discard >= 3:
                    score = 7000  # High value ACE
                elif total_discard >= 1:
                    score = 4000
                else:
                    score = -1

            elif cid == Energy_Retrieval:
                # Item: recover up to 2 Basic Energy from discard
                if discard_cnt.get(Basic_L_Energy, 0) >= 1:
                    score = 4500
                else:
                    score = -1

            else:
                score = 5000  # Unknown trainer — play it

        return score

    # ---- ATTACH (energy or tool to Pokemon) ----
    if o.type == OptionType.ATTACH:
        card = _get_card(obs, AreaType.HAND, o.index, my_idx)
        if card is None:
            return 0

        # Find the target Pokemon
        target = _get_card(obs, o.inPlayArea, o.inPlayIndex, my_idx)
        if target is None or not isinstance(target, Pokemon):
            return 0

        if card.id == Basic_L_Energy:
            # Manual energy attach (1 per turn) — use ENERGY_IDEAL priorities
            score = _attacker_energy_score(target, False, o.inPlayArea == AreaType.ACTIVE,
                                           opp_fighting=opp_fighting, opp_crustle=opp_crustle)
            # Bonus for attacker that needs energy
            if o.inPlayArea == AreaType.ACTIVE:
                if _plan.attacker_idx == 0 and _plan.needs_energy:
                    score += 300
            else:
                if _plan.attacker_idx == o.inPlayIndex + 1 and _plan.needs_energy:
                    score += 300
        else:
            score = 5000  # Other tool/card attachment

        return score

    # ---- EVOLVE ----
    if o.type == OptionType.EVOLVE:
        poke = _get_card(obs, o.inPlayArea, o.inPlayIndex, my_idx)
        if poke is None:
            return -1

        card = _get_card(obs, AreaType.HAND, o.index, my_idx)
        if card is None:
            return -1

        score = 9000 + len(poke.energies) * 10

        if card.id == Iono_Bellibolt:
            # Always evolve Tadbulb → Bellibolt ex (critical for ability + 280HP body)
            score += 2000
            if opp_fighting:
                score += 1000  # Even more urgent vs Fighting — 280HP body is our lifeline
        elif card.id == Iono_Kilowattrel:
            # Evolve Wattrel → Kilowattrel (draw ability)
            score += 1000

        # Don't evolve if it would strand our active attacker during planned attack
        if o.inPlayArea == AreaType.ACTIVE and _plan.attacker_idx == 0:
            # Evolving active resets energy & can't attack same turn — skip if we can attack now
            score -= 500

        return score

    # ---- ABILITY ----
    if o.type == OptionType.ABILITY:
        card = _get_card(obs, o.area, o.index, my_idx)
        if card is None:
            return 0

        if card.id == Iono_Bellibolt:
            # Electric Streamer: attach L from hand to any Iono Pokemon
            # Use it if we have L energies in hand
            l_in_hand = sum(1 for c in my.hand if c.id == Basic_L_Energy)
            if l_in_hand > 0:
                return 35000  # Highest priority — this drives the engine
            else:
                return -1

        elif card.id == Iono_Kilowattrel:
            # Flashing Draw: discard 1 L → draw to 6
            # Use when hand is thin AND Kilowattrel has a L attached
            kilo_poke = _get_card(obs, o.area, o.index, my_idx)
            has_l = isinstance(kilo_poke, Pokemon) and len(kilo_poke.energies) >= 1
            if has_l and len(my.hand) <= 4:
                return 25000
            elif has_l and len(my.hand) <= 5:
                return 15000
            else:
                return -1

        elif card.id == Levincia:
            # Stadium ability: put up to 2 L from discard to hand
            if discard_cnt.get(Basic_L_Energy, 0) >= 1 and not state.supporterPlayed:
                return 20000
            elif discard_cnt.get(Basic_L_Energy, 0) >= 1:
                return 15000
            else:
                return -1

        return 10000  # Unknown ability

    # ---- RETREAT ----
    if o.type == OptionType.RETREAT:
        active = my.active[0] if my.active else None
        if active is None:
            return -1

        # Retreat if: active can't attack, or there's a much better attacker on bench.
        # Guard: if the active CAN already attack this turn (plan targets active = idx 0),
        # don't retreat — that would waste the turn.
        if _plan.attacker_idx == 0:
            return -1  # Planned attacker is already active — no need to retreat

        if _plan.attacker_idx >= 1:
            # Guard: only retreat if the planned bench attacker is different from active
            # (prevents retreat loops after switching already happened)
            bench = [p for p in (my.bench or []) if p is not None]
            bench_idx = _plan.attacker_idx - 1
            if bench_idx < len(bench):
                planned_poke = bench[bench_idx]
                if planned_poke.id != active.id:
                    return 3000  # Retreat to the planned bench attacker
            return -1  # Bench index out of range or same Pokemon — don't retreat

        if active.hp <= 30:
            return 2000  # Active is nearly KO'd — retreat
        elif active.id not in IONO_POKEMON:
            return 1500  # Non-Iono active — get out
        elif opp_fighting and active.id in (Iono_Voltorb, Iono_Tadbulb, Iono_Wattrel):
            # Against Fighting: fragile basics (60-70HP) get 1-shot at weakness.
            # Retreat to Bellibolt ex if it has enough energy to attack (>=4) on bench.
            bellibolt_ready = any(
                p is not None and p.id == Iono_Bellibolt and len(p.energies) >= 4
                for p in (my.bench or [])
            )
            kilowattrel_ready = any(
                p is not None and p.id == Iono_Kilowattrel and len(p.energies) >= 3
                for p in (my.bench or [])
            )
            if bellibolt_ready or kilowattrel_ready:
                return 2500  # Bring up the ready tanky attacker
        return -1  # Stay

    # ---- ATTACK ----
    if o.type == OptionType.ATTACK:
        # The Voltorb Voltaic Chain strategy benefits from charging up (delaying attack)
        # because damage scales with total L-energy across all Iono Pokemon. So we keep
        # the base non-lethal attack score LOW (1100) — let setup actions win normally.
        # The CRITICAL fix: a LETHAL attack must ALWAYS fire, even over Bellibolt ability
        # (35000). Previously lethal attacks were stuck at 1100 and never fired → games
        # dragged 200+ turns with KO opportunities wasted every turn.
        score = 1000
        if _plan.attacker_idx >= 0 and not _plan.needs_energy:
            score += 100  # Charged: slight boost (original +100 for attack_idx==0)

        # Prefer the planned attack index
        if _plan.attack_idx == 0 and o.attackId is not None:
            score += 100

        # LETHAL bonus — a KO available this turn must outrank ALL setup actions
        # (including Bellibolt ability at 35000). This is the primary fix.
        # Guard: attacker_idx >= 0 ensures _plan was actually computed this turn.
        if _plan.attacker_idx >= 0 and _plan.lethal_dmg >= 0:
            score += 50000  # Lethal → game-winning priority

        return score

    return score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _agent_impl(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _deck_ids()
    return _decide(obs)


def _deck_ids() -> list[int]:
    """Return the Iono 60-card deck."""
    return [
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


def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point — compatible with gate_runner instrumentation."""
    global _n_model_decisions, _n_trivial_fallback

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        _n_model_decisions += 1
        return _deck_ids()

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
    max_count = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    try:
        result = _agent_impl(obs_dict)
        if (result
                and all(0 <= i < n_options for i in result)
                and len(result) >= min_count
                and len(set(result)) == len(result)):
            _n_model_decisions += 1
            return result
    except Exception:
        pass

    # Trivial legal fallback
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
