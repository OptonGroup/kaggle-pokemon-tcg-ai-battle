"""v003_search — Mega Lucario ex + determinized engine-search agent.

Self-contained for Kaggle submission: imports only from local cg/ + stdlib.
No src/ imports.

Strategy:
  - At MAIN decisions with >=2 options, run determinized 1-ply + rollout search:
      * sample K plausible hidden states (determinizations),
      * for each determinization, for each legal root option, search_step into it
        and roll out to terminal with the heuristic pilot (or to a depth cap),
      * aggregate scores; pick the best root option.
  - All other decisions: heuristic directly.
  - Time budget: tracks time spent; per-decision budget = remaining / est. decisions.
    If budget tiny, skip search and use heuristic.
  - NEVER throws: public agent() wraps everything with try/except fallback.
"""
import math
import os
import sys
import time
import random
from collections import defaultdict

# Ensure local cg/ is importable (both from submission dir and Kaggle runtime)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_card_data, to_observation_class,
    search_begin, search_step, search_end,
)

# ===========================================================================
# Deck loading
# ===========================================================================

def _load_deck() -> list[int]:
    candidates = [
        os.path.join(_HERE, "deck.csv"),
        "deck.csv",
        "/kaggle_simulations/agent/deck.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.read().split("\n")
            deck = []
            for line in lines:
                line = line.strip()
                if line:
                    deck.append(int(line))
                if len(deck) == 60:
                    break
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("Could not find deck.csv.")


my_deck = _load_deck()

# Card metadata
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist card IDs
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


# ===========================================================================
# Heuristic agent (inlined from src/agents/heuristic_agent.py)
# ===========================================================================

class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0
ability_used = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> "Pokemon | Card | None":
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


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
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


def decide_on_obs(obs: Observation) -> list[int]:
    """Heuristic decision on an already-parsed Observation dataclass."""
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

    global plan, pre_turn, ability_used
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False

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
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
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
                        data = card_table[op_pokemon.id]
                        if data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif data.resistance == EnergyType.FIGHTING:
                            damage -= 30
                        prize = 0
                        score = pokemon_score(op_pokemon)
                        if op_pokemon.hp <= damage:
                            prize = prize_count(op_pokemon)
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
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = op_pokemon.hp - damage
                            plan.energy = more_energy

    def energy_score(pokemon: Pokemon, active: bool) -> int:
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

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
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
                        if o.index == plan.target - 1:
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
                        if not ability_used or not state.energyAttached:
                            score += 30
                        else:
                            score -= 1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
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
                    if plan.attacker <= 0:
                        score = -1
                    else:
                        score = 6000
                elif card.id == Premium_Power_Pro:
                    if state.supporterPlayed and plan.remain_hp <= 0:
                        score = -1
                    elif not can_attack:
                        if not state.supporterPlayed and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                            score = 3050
                        else:
                            score = -1
                    else:
                        score = 5000
                elif card.id == Boss_Orders:
                    if plan.target >= 1:
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
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == 1267:
                score = 1
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = 2000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Lunatone:
                ability_used = True
    return desc_indices[:select.maxCount]


def _heuristic_impl(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return decide_on_obs(obs)


# ===========================================================================
# Search agent config
# Env-var overrides: V003_USABLE_FRACTION, V003_MAX_DECISION_S, V003_MIN_SIMS,
# V003_DET_BATCH. Useful for local gate (smaller budget) vs Kaggle (full budget).
# ===========================================================================

TOTAL_BANK_S = 600.0
USABLE_FRACTION = float(os.environ.get("V003_USABLE_FRACTION", "0.55"))
EST_DECISIONS = 120            # conservative est. of our decisions in a long game
MIN_BUDGET_S = 0.10            # below this per-decision budget, skip search
HARD_PER_DECISION_CAP_S = float(os.environ.get("V003_MAX_DECISION_S", "6.0"))
MIN_SIMS = int(os.environ.get("V003_MIN_SIMS", "10"))  # min rollouts per option before trusting UCB1
DET_BATCH = int(os.environ.get("V003_DET_BATCH", "5"))  # rollouts between fresh determinizations
ROLLOUT_DEPTH_CAP = 140        # max search_steps per rollout before leaf eval
UCB_C = 1.414                  # UCB1 exploration constant

# Safe Basic Pokemon IDs for padding / face-down opponent active (all from Lucario list)
_SAFE_BASIC = 677   # Riolu
_PRIOR_BASICS = (677, 676, 675, 673)  # Riolu, Solrock, Lunatone, Makuhita

# ===========================================================================
# Per-game time tracking
# ===========================================================================

_time_spent = 0.0
_decisions_made = 0
_game_started = False


def _reset_game_state():
    global _time_spent, _decisions_made, _game_started
    _time_spent = 0.0
    _decisions_made = 0
    _game_started = True


# ===========================================================================
# Determinization helpers
# ===========================================================================

def _multiset_subtract(pool, remove):
    out = list(pool)
    for r in remove:
        if r in out:
            out.remove(r)
    return out


def _pad_to(lst, n, fill):
    out = list(lst)
    while len(out) < n:
        out.append(fill)
    return out[:n]


def _visible_card_ids(player_state, stadium, player_index) -> list[int]:
    seen = []
    if player_state.hand:
        for c in player_state.hand:
            seen.append(c.id)
    for pk in list(player_state.active) + list(player_state.bench):
        if pk is None:
            continue
        seen.append(pk.id)
        for c in pk.energyCards:
            seen.append(c.id)
        for c in pk.tools:
            seen.append(c.id)
        for c in pk.preEvolution:
            seen.append(c.id)
    for c in player_state.discard:
        seen.append(c.id)
    for c in player_state.prize:
        if c is not None:
            seen.append(c.id)
    for c in stadium:
        if c.playerIndex == player_index:
            seen.append(c.id)
    return seen


def build_determinization(obs: Observation, rng: random.Random):
    """Sample (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
    with sizes matching the engine's requirements.

    Our hidden pool = full 60-card deck minus everything visible to us.
    Opponent prior = same Lucario list (v003 prior; gate opponent IS Lucario).
    """
    st = obs.current
    yi = st.yourIndex
    me = st.players[yi]
    opp = st.players[1 - yi]

    # Our hidden cards
    our_hidden = _multiset_subtract(list(my_deck), _visible_card_ids(me, st.stadium, yi))
    rng.shuffle(our_hidden)
    need_my_prize = len(me.prize)
    need_my_deck = me.deckCount
    our_hidden = _pad_to(our_hidden, need_my_prize + need_my_deck, _SAFE_BASIC)
    your_prize = our_hidden[:need_my_prize]
    your_deck = our_hidden[need_my_prize:need_my_prize + need_my_deck]

    # Opponent hidden cards (prior = our Lucario list)
    opp_visible = _visible_card_ids(opp, st.stadium, 1 - yi)
    opp_hidden = _multiset_subtract(list(my_deck), opp_visible)
    rng.shuffle(opp_hidden)
    need_opp_prize = len(opp.prize)
    need_opp_deck = opp.deckCount
    need_opp_hand = opp.handCount
    total_opp = need_opp_prize + need_opp_deck + need_opp_hand
    opp_hidden = _pad_to(opp_hidden, total_opp, _SAFE_BASIC)
    opp_prize = opp_hidden[:need_opp_prize]
    rest = opp_hidden[need_opp_prize:]
    opp_deck = rest[:need_opp_deck]
    rest2 = rest[need_opp_deck:]
    opp_hand = rest2[:need_opp_hand]

    # Face-down opponent active
    opp_active = []
    if len(opp.active) > 0 and opp.active[0] is None:
        opp_active = [_PRIOR_BASICS[0]]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


# ===========================================================================
# Leaf eval + rollout
# ===========================================================================

def _leaf_eval(obs: Observation, our_index: int) -> float:
    """Heuristic value at a non-terminal leaf, in [-1, 1]."""
    st = obs.current
    me = st.players[our_index]
    opp = st.players[1 - our_index]
    my_prize = len(me.prize)
    opp_prize = len(opp.prize)
    # Prize differential: fewer prizes = closer to win; range [-6,6] → normalize
    v = (opp_prize - my_prize) / 6.0 * 0.8

    def board_strength(ps):
        s = 0
        for pk in list(ps.active) + list(ps.bench):
            if pk is None:
                continue
            s += pk.hp + len(pk.energies) * 20
        return s

    my_b = board_strength(me)
    opp_b = board_strength(opp)
    denom = (my_b + opp_b) or 1
    v += ((my_b - opp_b) / denom) * 0.2
    return max(-1.0, min(1.0, v))


def _result_value(result: int, our_index: int) -> float:
    if result == our_index:
        return 1.0
    if result == (1 - our_index):
        return -1.0
    return 0.0  # draw


def _legalize_choice(choice, sel):
    """Coerce a heuristic choice into bounds [minCount, maxCount], no dups."""
    n = len(sel.option)
    seen = []
    for c in choice:
        if isinstance(c, int) and 0 <= c < n and c not in seen:
            seen.append(c)
    if len(seen) > sel.maxCount:
        seen = seen[:sel.maxCount]
    if len(seen) < sel.minCount:
        for i in range(n):
            if i not in seen:
                seen.append(i)
            if len(seen) >= sel.minCount:
                break
    return seen


def _rollout_choice(obs: Observation, rng: random.Random):
    """Pick a move for a rollout step; fallback to first legal indices."""
    sel = obs.select
    try:
        choice = decide_on_obs(obs)
        return _legalize_choice(choice, sel)
    except Exception:
        n = len(sel.option)
        k = max(sel.minCount, 0)
        return list(range(min(k, n)))


def _rollout_from(state, our_index: int, rng: random.Random, deadline: float) -> float:
    """Roll out a SearchState to terminal (or depth/time cap). Returns value in [-1,1]."""
    steps = 0
    while True:
        obs = state.observation
        if obs.current is not None and obs.current.result != -1:
            return _result_value(obs.current.result, our_index)
        sel = obs.select
        if sel is None:
            if obs.current is not None and obs.current.result != -1:
                return _result_value(obs.current.result, our_index)
            return _leaf_eval(obs, our_index)
        if steps >= ROLLOUT_DEPTH_CAP or time.perf_counter() >= deadline:
            return _leaf_eval(obs, our_index)
        choice = _rollout_choice(obs, rng)
        state = search_step(state.searchId, choice)
        steps += 1


# ===========================================================================
# Budget tracking
# ===========================================================================

def _per_decision_budget() -> float:
    remaining = TOTAL_BANK_S * USABLE_FRACTION - _time_spent
    if remaining <= 0:
        return 0.0
    est_left = max(1, EST_DECISIONS - _decisions_made)
    budget = remaining / est_left
    return min(budget, HARD_PER_DECISION_CAP_S)


# ===========================================================================
# Core search decision
# ===========================================================================

def _save_heuristic_globals():
    return (plan, pre_turn, ability_used)


def _restore_heuristic_globals(saved):
    global plan, pre_turn, ability_used
    plan, pre_turn, ability_used = saved


def _ucb1_select(scores, counts, total_n, n_opts):
    """Select the option index with highest UCB1 value.
    Options with 0 counts get infinite priority (ensure each visited at least once)."""
    best_i = -1
    best_v = -1e18
    for i in range(n_opts):
        if counts[i] == 0:
            return i  # visit unvisited options first
        mean = scores[i] / counts[i]
        ucb = mean + UCB_C * math.sqrt(math.log(total_n) / counts[i])
        if ucb > best_v:
            best_v = ucb
            best_i = i
    return best_i


def _search_decision(obs: Observation, budget: float, rng: random.Random):
    """Run budget-driven determinized rollout search for a MAIN single-select decision.

    Algorithm:
      - Keep a determinization open for DET_BATCH rollouts, then rotate to a fresh one.
      - Select options via round-robin until each has MIN_SIMS, then switch to UCB1
        to focus on promising options while continuing to explore.
      - Returns [best_option_index] (highest mean value) or None if search failed.
    """
    sel = obs.select
    n_opts = len(sel.option)
    if n_opts == 0:
        return None
    # Only search single-select MAIN decisions
    if sel.minCount != 1 or sel.maxCount != 1:
        return None

    our_index = obs.current.yourIndex
    deadline = time.perf_counter() + budget

    scores = [0.0] * n_opts
    counts = [0] * n_opts
    total_rollouts = 0

    current_root = None
    det_uses = 0

    def open_new_det():
        nonlocal current_root, det_uses
        if current_root is not None:
            try:
                search_end()
            except Exception:
                pass
            current_root = None
        if time.perf_counter() >= deadline:
            return False
        det = build_determinization(obs, rng)
        yd, yp, od, op_p, oh, oa = det
        try:
            root = search_begin(obs, yd, yp, od, op_p, oh, oa)
            current_root = root
            det_uses = 0
            return True
        except Exception:
            try:
                search_end()
            except Exception:
                pass
            return False

    def close_det():
        nonlocal current_root
        if current_root is not None:
            try:
                search_end()
            except Exception:
                pass
            current_root = None

    saved = _save_heuristic_globals()
    try:
        while time.perf_counter() < deadline:
            if current_root is None:
                if not open_new_det():
                    break

            # Select option: round-robin until all have MIN_SIMS, then UCB1
            all_have_min = all(c >= MIN_SIMS for c in counts)
            if not all_have_min:
                opt_idx = min(range(n_opts), key=lambda i: counts[i])
            else:
                opt_idx = _ucb1_select(scores, counts, total_rollouts, n_opts)

            if time.perf_counter() >= deadline:
                break

            try:
                child = search_step(current_root.searchId, [opt_idx])
            except Exception:
                close_det()
                continue

            obs_c = child.observation
            if obs_c.current is not None and obs_c.current.result != -1:
                val = _result_value(obs_c.current.result, our_index)
            else:
                val = _rollout_from(child, our_index, rng, deadline)

            scores[opt_idx] += val
            counts[opt_idx] += 1
            total_rollouts += 1
            det_uses += 1

            if det_uses >= DET_BATCH:
                close_det()
    finally:
        close_det()
        _restore_heuristic_globals(saved)

    evaluated = [i for i in range(n_opts) if counts[i] > 0]
    if not evaluated:
        return None
    best_i = max(evaluated, key=lambda i: scores[i] / counts[i])
    return [best_i]


# ===========================================================================
# Public entry point
# ===========================================================================

def _agent_impl(obs_dict: dict) -> list[int]:
    """Core agent logic (may raise; wrapped by agent())."""
    global _time_spent, _decisions_made, _game_started

    obs = to_observation_class(obs_dict)

    # Deck selection → fresh game
    if obs.select is None:
        _reset_game_state()
        return my_deck

    if not _game_started:
        _reset_game_state()

    sel = obs.select
    context = sel.context

    t0 = time.perf_counter()
    try:
        use_search = (
            context == SelectContext.MAIN
            and len(sel.option) >= 2
            and sel.minCount == 1 and sel.maxCount == 1
            and obs_dict.get("search_begin_input") is not None
        )
        budget = _per_decision_budget() if use_search else 0.0

        if use_search and budget >= MIN_BUDGET_S:
            rng = random.Random((_decisions_made * 1000003) ^ 0x9E3779B1)
            chosen = _search_decision(obs, budget, rng)
            if chosen is not None:
                return chosen

        # Fallback / non-MAIN: heuristic
        return _heuristic_impl(obs_dict)
    finally:
        dt = time.perf_counter() - t0
        _time_spent += dt
        _decisions_made += 1


def agent(obs_dict: dict) -> list[int]:
    """Never-throw public wrapper.

    Falls back to heuristic, then to trivially-legal indices on any exception.
    """
    try:
        return _agent_impl(obs_dict)
    except Exception:
        try:
            return _heuristic_impl(obs_dict)
        except Exception:
            try:
                obs = to_observation_class(obs_dict)
                if obs.select is None:
                    return my_deck
                n = len(obs.select.option)
                k = max(1, obs.select.minCount)
                return list(range(min(k, n)))
            except Exception:
                return []
