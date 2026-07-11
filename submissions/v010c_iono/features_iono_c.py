"""
features_iono.py — Per-option feature matrix for the Iono Electric Swarm IL ranker.

Works entirely from raw obs_dict (same format as live agent observation and replay JSON).
No libcg required; all card metadata is loaded from data/EN_Card_Data.csv.

Deck archetype: Iono's Electric Swarm
  - Voltorb (265) + Bellibolt ex (269) energy accel + Kilowattrel (271) draw engine
  - Voltorb's "Voltaic Chain" does 20 + 20× {L} energy on ALL my Pokemon — scales with board state

Feature layout (per row = one option), total FEATURE_DIM = 66:
  [0..16]   Option type one-hot (17 types: NUMBER=0 .. SPECIAL_CONDITION=16)
  --- Local per-option features (slots 17..42, 26 slots) ---
  [17]  ATTACK: damage
  [18]  ATTACK: target current HP
  [19]  ATTACK: is-KO flag (damage >= target_hp)
  [20]  ATTACK: target prize value (1=normal, 2=ex, 3=megaEx)
  [21]  ATTACK: weakness match (1 if attacker type == target weakness)
  [22]  PLAY/ATTACH/EVOLVE/CARD: card_type int (0-6)
  [23]  PLAY/ATTACH/EVOLVE/CARD: is_supporter flag
  [24]  PLAY/ATTACH/EVOLVE/CARD: is_item flag
  [25]  PLAY/ATTACH/EVOLVE/CARD: is_pokemon flag
  [26]  PLAY/ATTACH/EVOLVE/CARD: is_energy flag
  [27]  PLAY/ATTACH/EVOLVE/CARD: target pokemon energy count
  [28]  PLAY/ATTACH/EVOLVE/CARD: card HP
  [29]  PLAY/ATTACH/EVOLVE/CARD: card retreat cost
  [30]  RETREAT: own active HP ratio (hp/maxhp)
  [31]  RETREAT: best bench HP
  [32]  END: is_end flag
  [33]  ATTACK: effective damage after weakness (damage*2 if weakness_match else damage)
  [34]  ATTACK: KO after weakness (effective_damage >= opp_active_hp)
  [35]  PLAY/EVOLVE/CARD: card_is_bellibolt_ex (card id==269)
  [36]  PLAY/EVOLVE/CARD: card_is_voltorb (card id==265)
  [37]  ATTACH: attach_target_is_voltorb (inPlayArea pokemon id==265) [FIXED: was dead retreat_to_kilowattrel]
  [38]  ATTACH: attach_target_is_bellibolt (inPlayArea pokemon id==269)
  [39]  ATTACH: attach_energy_is_lightning (energy card has {L} type)
  [40]  ATTACH: attach_to_ready_attacker (target is Voltorb/Bellibolt with >=1 energy already)
  [41]  ATTACK: voltaic_chain_would_ko (20 + 20*total_lightning >= opp_active_hp, active is Voltorb)
  [42]  global: n_voltorb_on_bench (count of id265 on bench, raw count)
  --- Global state features (slots 43..65, 23 values) ---
  [43]  turn / 100 (normalized)
  [44]  my prizes left
  [45]  opp prizes left
  [46]  my hand size
  [47]  my bench count
  [48]  opp bench count
  [49]  my active HP
  [50]  my active maxHP
  [51]  opp active HP
  [52]  opp active maxHP
  [53]  energy on my active
  [54]  supporter_played flag
  [55]  energy_attached flag
  [56]  retreated flag
  [57]  n_options in this decision
  [58]  select_context (raw int)
  [59]  prize_diff = opp_prizes_remaining - my_prizes_remaining  (positive = we're ahead)
  [60]  opp_active_hp_ratio = opp_active_hp / max(1, opp_active_maxhp)
  [61]  my_bench_energy_total = sum of energies count across my bench pokemon
  [62]  total_lightning_on_board = count of {L} energy (type 4) across all my Pokemon (active + bench)
  [63]  n_bellibolt_in_play = count of id269 among my active+bench, normalized /3
  [64]  my_active_is_voltorb = 1 if my active pokemon id==265
  [65]  n_my_pokemon_in_play = count of my Pokemon in play (active+bench), normalized /6

Total: 66 features per option.

NOTE: atk_effective_damage (slot 33) reports the base damage from EN_Card_Data.csv.
Voltorb's 'Voltaic Chain' damage scales with board lightning count (slot 57) which is
NOT captured in the static CSV damage — model must learn the interaction via slot[57].

Obs field access notes:
  obs['select']['option']: list of option dicts
  obs['current']['yourIndex']: int (0 or 1)
  obs['current']['players'][i]: PlayerState dict
  obs['current']['players'][i]['active']: list (0 or 1 items) of pokemon dicts
  obs['current']['players'][i]['bench']: list of pokemon dicts
  obs['current']['players'][i]['hand']: list of card dicts (own) or None (opp)
  obs['current']['players'][i]['handCount']: int
  obs['current']['players'][i]['prize']: list of card dicts
  obs['current']['supporterPlayed']: bool
  obs['current']['energyAttached']: bool
  obs['current']['retreated']: bool
  obs['current']['turn']: int
  pokemon dict: {id, hp, maxHp, energies, energyCards, tools, ...}
  option dict: {type, [area, index, playerIndex, inPlayArea, inPlayIndex, attackId, ...]}
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

N_OPT_TYPES = 17   # slots 0-16
N_LOCAL = 26       # slots 17-42  (was 21, +5 new combo-execution per-option features)
N_GLOBAL = 23      # slots 43-65  (unchanged 23 global features)
FEATURE_DIM = N_OPT_TYPES + N_LOCAL + N_GLOBAL  # = 66

# Iono's Electric Swarm deck card IDs
VOLTORB_ID = 265
TADBULB_ID = 268
BELLIBOLT_EX_ID = 269
WATTREL_ID = 270
KILOWATTREL_ID = 271
LIGHTNING_ENERGY_TYPE = 4  # {L} energy type int

# OptionType ints (mirrors cg/api.py OptionType)
OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15
OPT_SPECIAL_CONDITION = 16

# Energy type string → int (matches EnergyType enum)
_ENERGY_STR_MAP = {
    "{C}": 0, "{G}": 1, "{R}": 2, "{W}": 3, "{L}": 4,
    "{P}": 5, "{F}": 6, "{D}": 7, "{M}": 8, "{N}": 9,
    "{A}": 10,
}

_DATA_PATH = Path(__file__).parent / "EN_Card_Data.csv"


def _parse_energy_str(s: str) -> int | None:
    """Parse a single energy token like '{F}' → int (EnergyType). Returns None if unknown."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    return _ENERGY_STR_MAP.get(s)


def _parse_damage(s: Any) -> int:
    """Parse damage string like '130', '130+', '130x', or NaN → int."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0
    try:
        return int(re.sub(r"[^0-9]", "", str(s)) or "0")
    except Exception:
        return 0


@lru_cache(maxsize=1)
def _load_card_db() -> dict[int, dict]:
    """
    Load card metadata from EN_Card_Data.csv.
    Returns card_info: card_id → dict of card properties.
    """
    df = pd.read_csv(_DATA_PATH)
    df.columns = df.columns.str.strip()

    stage_col = "Stage (Pokémon)/Type (Energy and Trainer)"

    card_info: dict[int, dict] = {}
    for card_id, grp in df.groupby("Card ID"):
        row = grp.iloc[0]
        stage = str(row.get(stage_col, "") or "")
        is_pokemon = "Pokémon" in stage
        is_basic_energy = "Basic Energy" in stage
        is_special_energy = "Special Energy" in stage
        is_supporter = stage.strip() == "Supporter"
        is_item = stage.strip() == "Item"
        is_tool = "Tool" in stage
        is_energy = is_basic_energy or is_special_energy
        is_stadium = "Stadium" in stage

        # CardType int: POKEMON=0, ITEM=1, TOOL=2, SUPPORTER=3, STADIUM=4, BASIC_ENERGY=5, SPECIAL_ENERGY=6
        if is_pokemon:
            ct = 0
        elif is_item:
            ct = 1
        elif is_tool:
            ct = 2
        elif is_supporter:
            ct = 3
        elif is_stadium:
            ct = 4
        elif is_basic_energy:
            ct = 5
        elif is_special_energy:
            ct = 6
        else:
            ct = 0

        hp_val = row.get("HP")
        try:
            hp = int(float(hp_val)) if hp_val is not None and not (isinstance(hp_val, float) and np.isnan(hp_val)) else 0
        except (ValueError, TypeError):
            hp = 0

        ret_val = row.get("Retreat")
        try:
            retreat = int(float(ret_val)) if ret_val is not None and not (isinstance(ret_val, float) and np.isnan(ret_val)) else 0
        except (ValueError, TypeError):
            retreat = 0

        weak_str = str(row.get("Weakness") or "")
        weak_e = _parse_energy_str(weak_str)

        rule = str(row.get("Rule") or "").lower()
        name_str = str(row.get("Card Name") or "").lower()
        is_ex = ("ex rule" in rule) or (name_str.endswith(" ex") and is_pokemon)
        is_mega_ex = ("mega evolution" in rule) or (name_str.startswith("mega ") and is_pokemon)

        type_str = str(row.get("Type") or "")
        type_match = re.search(r"\{[A-Za-z]+\}", type_str)
        own_energy = _parse_energy_str(type_match.group()) if type_match else None

        # Build attack damage list (one entry per row/attack of this card)
        damages = [_parse_damage(r.get("Damage")) for _, r in grp.iterrows()]

        card_info[int(card_id)] = {
            "hp": hp,
            "retreat": retreat,
            "is_pokemon": is_pokemon,
            "is_supporter": is_supporter,
            "is_item": is_item,
            "is_tool": is_tool,
            "is_energy": is_energy,
            "card_type_int": ct,
            "weakness_energy": weak_e,
            "own_energy": own_energy,
            "is_ex": is_ex,
            "is_mega_ex": is_mega_ex,
            "damages": damages,
        }

    return card_info


def _get_card(card_id: int) -> dict:
    """Safe card lookup; returns empty dict if not found."""
    db = _load_card_db()
    return db.get(int(card_id) if card_id else 0, {})


def _get_pokemon_from_obs(player_dict: dict, area: int, idx: int) -> dict | None:
    """
    Resolve a pokemon from the player state dict using area + index.
    area: 4=ACTIVE, 5=BENCH
    """
    if area == 4:  # ACTIVE
        active = player_dict.get("active") or []
        return active[0] if active else None
    elif area == 5:  # BENCH
        bench = player_dict.get("bench") or []
        return bench[idx] if idx < len(bench) else None
    return None


def build_feature_matrix(obs_dict: dict) -> np.ndarray:
    """
    Build per-option feature matrix from a raw obs_dict.

    Args:
        obs_dict: Raw observation dict from replay or live agent (same schema).

    Returns:
        X: np.ndarray of shape (n_options, FEATURE_DIM=61), float32.
           Rows correspond to obs_dict['select']['option'] in order.
           Never raises; all field access is fully defensive.
    """
    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    n_opts = len(opts)
    ctx = int(sel.get("context") or 0)

    if n_opts == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    cur = obs_dict.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    players = cur.get("players") or [{}, {}]
    while len(players) < 2:
        players = players + [{}]

    my_pl = players[me]
    opp_pl = players[1 - me]

    # Active pokemon dicts
    my_active = (my_pl.get("active") or [None])[0]
    opp_active = (opp_pl.get("active") or [None])[0]

    my_active_hp = int((my_active or {}).get("hp") or 0)
    my_active_maxhp = int((my_active or {}).get("maxHp") or 1) or 1
    opp_active_hp = int((opp_active or {}).get("hp") or 0)
    opp_active_maxhp = int((opp_active or {}).get("maxHp") or 1) or 1

    my_energy_count = len((my_active or {}).get("energies") or [])

    my_prizes = len(my_pl.get("prize") or [])
    opp_prizes = len(opp_pl.get("prize") or [])
    my_hand_count = int(my_pl.get("handCount") or 0)
    my_bench_list = my_pl.get("bench") or []
    my_bench = len(my_bench_list)
    opp_bench = len(opp_pl.get("bench") or [])

    turn = int(cur.get("turn") or 0)
    supporter_played = float(bool(cur.get("supporterPlayed")))
    energy_attached = float(bool(cur.get("energyAttached")))
    retreated = float(bool(cur.get("retreated")))

    # My active card metadata (for attacker energy type + attack damages)
    my_active_card_id = int((my_active or {}).get("id") or 0)
    my_card = _get_card(my_active_card_id)
    my_own_energy = my_card.get("own_energy")  # None or int

    # Opp active card metadata (for prize value + weakness)
    opp_active_card_id = int((opp_active or {}).get("id") or 0)
    opp_card = _get_card(opp_active_card_id)
    opp_weakness_energy = opp_card.get("weakness_energy")  # None or int
    opp_prize_val = 3.0 if opp_card.get("is_mega_ex") else (2.0 if opp_card.get("is_ex") else 1.0)

    # Hand cards (for PLAY / ATTACH index lookups)
    my_hand_cards = my_pl.get("hand") or []

    # ── New global features ───────────────────────────────────────────────────
    # prize_diff: opp_prizes_remaining - my_prizes_remaining
    # positive means we've taken more prizes (we're ahead)
    prize_diff = float(opp_prizes - my_prizes)

    # opp_active_hp_ratio
    opp_active_hp_ratio = float(opp_active_hp) / float(max(1, opp_active_maxhp))

    # my_bench_energy_total: sum of energies count across my bench pokemon
    my_bench_energy_total = float(sum(
        len((b or {}).get("energies") or []) for b in my_bench_list
    ))

    # total_lightning_on_board: count of {L} energy (type 4) across my active + all bench pokemon
    # NOTE: Voltorb's "Voltaic Chain" damage scales with this — the static CSV damage
    # in slot 33 does NOT capture the scaling; model must learn the interaction via this slot.
    _count_lightning = lambda pk: len([e for e in (pk or {}).get("energies") or [] if e == LIGHTNING_ENERGY_TYPE])
    total_lightning_on_board = float(
        _count_lightning(my_active) + sum(_count_lightning(b) for b in my_bench_list)
    )

    # n_bellibolt_in_play: count of Bellibolt ex (id269) among my active+bench, normalized /3
    n_bellibolt_in_play = float(
        sum(1 for pk in ([my_active] + list(my_bench_list))
            if pk and int((pk or {}).get("id") or 0) == BELLIBOLT_EX_ID) / 3.0
    )

    # my_active_is_voltorb: 1 if my active pokemon is Voltorb (id265)
    my_active_is_voltorb = float(my_active_card_id == VOLTORB_ID)

    # n_my_pokemon_in_play: count of my Pokemon in play (active+bench), normalized /6
    n_my_pokemon_in_play = float(
        len([p for p in ([my_active] + list(my_bench_list)) if p]) / 6.0
    )

    # n_voltorb_on_bench: count of Voltorbs (id265) on my bench (swarm width)
    # Used as LOCAL slot B+25 (same value for every option row in this decision)
    n_voltorb_on_bench = float(
        sum(1 for b in my_bench_list if b and int((b or {}).get("id") or 0) == VOLTORB_ID)
    )

    # Global vector (23 values = original 16 + 7 new)
    global_vec = np.array([
        turn / 100.0,
        float(my_prizes),
        float(opp_prizes),
        float(my_hand_count),
        float(my_bench),
        float(opp_bench),
        float(my_active_hp),
        float(my_active_maxhp),
        float(opp_active_hp),
        float(opp_active_maxhp),
        float(my_energy_count),
        supporter_played,
        energy_attached,
        retreated,
        float(n_opts),
        float(ctx),
        # new global features:
        prize_diff,
        opp_active_hp_ratio,
        my_bench_energy_total,
        total_lightning_on_board,
        n_bellibolt_in_play,
        my_active_is_voltorb,
        n_my_pokemon_in_play,
    ], dtype=np.float32)
    assert len(global_vec) == N_GLOBAL

    # Build the matrix
    X = np.zeros((n_opts, FEATURE_DIM), dtype=np.float32)

    for i, opt in enumerate(opts):
        row = X[i]
        opt_type = int(opt.get("type") or 0)

        # [0:17] one-hot option type
        if 0 <= opt_type < N_OPT_TYPES:
            row[opt_type] = 1.0

        # Local features: base offset = 17
        B = N_OPT_TYPES  # = 17

        if opt_type == OPT_ATTACK:
            # Determine attack rank (0-indexed) among ATTACK options in this select
            attack_rank = sum(1 for j in range(i) if int(opts[j].get("type") or 0) == OPT_ATTACK)
            damages = my_card.get("damages") or [0]
            damage = damages[attack_rank] if attack_rank < len(damages) else (damages[-1] if damages else 0)

            wk_match = float(
                my_own_energy is not None
                and opp_weakness_energy is not None
                and my_own_energy == opp_weakness_energy
            )

            row[B + 0] = float(damage)
            row[B + 1] = float(opp_active_hp)
            row[B + 2] = float(damage > 0 and damage >= opp_active_hp)
            row[B + 3] = opp_prize_val
            row[B + 4] = wk_match

            # New per-option attack features (slots B+16, B+17)
            eff_damage = float(damage * 2 if wk_match else damage)
            row[B + 16] = eff_damage
            row[B + 17] = float(eff_damage > 0 and eff_damage >= opp_active_hp)

            # [41] voltaic_chain_would_ko: Voltorb's Voltaic Chain KO signal
            # Voltaic Chain = 20 + 20 × (count of {L} energy on ALL my Pokemon)
            # total_lightning_on_board is computed below in global_vec (slot 62 = index 19 in global)
            # We compute inline here from the already-computed value.
            # Only fires when my active is Voltorb (id 265) — otherwise 0.
            if my_active_card_id == VOLTORB_ID:
                voltaic_dmg = 20 + 20 * int(total_lightning_on_board)
                row[B + 24] = float(voltaic_dmg > 0 and voltaic_dmg >= opp_active_hp)
            # else stays 0.0

        elif opt_type in (OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_CARD):
            card_id = 0
            target_pk: dict | None = None

            if opt_type == OPT_PLAY:
                hand_idx = int(opt.get("index") or 0)
                if hand_idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[hand_idx].get("id") or 0)

            elif opt_type == OPT_ATTACH:
                area = int(opt.get("area") or 0)
                idx = int(opt.get("index") or 0)
                if area == 2 and idx < len(my_hand_cards):  # from HAND
                    card_id = int(my_hand_cards[idx].get("id") or 0)
                # target pokemon
                in_area = int(opt.get("inPlayArea") or 0)
                in_idx = int(opt.get("inPlayIndex") or 0)
                target_pk = _get_pokemon_from_obs(my_pl, in_area, in_idx)

            elif opt_type == OPT_EVOLVE:
                area = int(opt.get("area") or 0)
                idx = int(opt.get("index") or 0)
                if area == 2 and idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[idx].get("id") or 0)

            elif opt_type == OPT_CARD:
                area = int(opt.get("area") or 0)
                pidx = int(opt.get("playerIndex") or 0)
                idx = int(opt.get("index") or 0)
                pl = players[pidx] if pidx < len(players) else {}
                if area == 2:  # HAND
                    h = pl.get("hand") or []
                    if idx < len(h):
                        card_id = int(h[idx].get("id") or 0)
                elif area == 4:  # ACTIVE
                    act = pl.get("active") or []
                    if act:
                        card_id = int(act[0].get("id") or 0)
                elif area == 5:  # BENCH
                    bn = pl.get("bench") or []
                    if idx < len(bn):
                        card_id = int(bn[idx].get("id") or 0)

            cd = _get_card(card_id) if card_id else {}
            row[B + 5] = float(cd.get("card_type_int") or 0)
            row[B + 6] = float(bool(cd.get("is_supporter")))
            row[B + 7] = float(bool(cd.get("is_item")))
            row[B + 8] = float(bool(cd.get("is_pokemon")))
            row[B + 9] = float(bool(cd.get("is_energy")))
            row[B + 10] = float(len((target_pk or {}).get("energies") or []))
            row[B + 11] = float(cd.get("hp") or 0)
            row[B + 12] = float(cd.get("retreat") or 0)

            # New per-option card identity features (slots B+18, B+19)
            # [35] card_is_bellibolt_ex: float(card_id == 269)
            # [36] card_is_voltorb: float(card_id == 265)
            row[B + 18] = float(card_id == BELLIBOLT_EX_ID)
            row[B + 19] = float(card_id == VOLTORB_ID)

            # New combo-execution ATTACH features (slots B+20..B+23)
            # Only meaningful for OPT_ATTACH; safe (default 0) for OPT_PLAY/OPT_EVOLVE/OPT_CARD.
            if opt_type == OPT_ATTACH:
                target_id = int((target_pk or {}).get("id") or 0)
                # [37] attach_target_is_voltorb: attaching TO a Voltorb (key combo target)
                row[B + 20] = float(target_id == VOLTORB_ID)
                # [38] attach_target_is_bellibolt: attaching TO a Bellibolt ex
                row[B + 21] = float(target_id == BELLIBOLT_EX_ID)
                # [39] attach_energy_is_lightning: the energy card being attached is {L}
                attached_card = _get_card(card_id) if card_id else {}
                attached_energy_type = attached_card.get("own_energy")
                row[B + 22] = float(attached_energy_type == LIGHTNING_ENERGY_TYPE)
                # [40] attach_to_ready_attacker: target is Voltorb/Bellibolt AND already has >=1 energy
                is_key_attacker = target_id in (VOLTORB_ID, BELLIBOLT_EX_ID)
                target_energy_count = len((target_pk or {}).get("energies") or [])
                row[B + 23] = float(is_key_attacker and target_energy_count >= 1)

        elif opt_type == OPT_RETREAT:
            ratio = my_active_hp / my_active_maxhp
            row[B + 13] = ratio
            bench_list = my_pl.get("bench") or []
            best_bench_hp = max((b.get("hp") or 0 for b in bench_list), default=0)
            row[B + 14] = float(best_bench_hp)
            # NOTE: RETREAT options have type=12 but NO bench-target fields in the option dict
            # (bench target is picked in a separate sub-select after retreat is confirmed).
            # Slots B+20..B+24 are filled below for OPT_ATTACH only.

        elif opt_type == OPT_END:
            row[B + 15] = 1.0

        # [42] n_voltorb_on_bench: count of Voltorbs (id265) on my bench (swarm-width signal)
        # This is a global state value assigned for every option row (always same within a decision).
        row[B + 25] = n_voltorb_on_bench

        # [43:66] global features
        row[N_OPT_TYPES + N_LOCAL : N_OPT_TYPES + N_LOCAL + N_GLOBAL] = global_vec

    return X


# Feature names for interpretability
FEATURE_NAMES: list[str] = [
    # one-hot type [0:17]
    "type_NUMBER", "type_YES", "type_NO", "type_CARD", "type_TOOL_CARD",
    "type_ENERGY_CARD", "type_ENERGY", "type_PLAY", "type_ATTACH", "type_EVOLVE",
    "type_ABILITY", "type_DISCARD", "type_RETREAT", "type_ATTACK", "type_END",
    "type_SKILL", "type_SPECIAL_CONDITION",
    # local attack [17:22]
    "atk_damage", "atk_target_hp", "atk_is_ko", "atk_prize_val", "atk_weakness_match",
    # local card [22:30]
    "card_type_int", "card_is_supporter", "card_is_item", "card_is_pokemon", "card_is_energy",
    "card_target_energy_count", "card_hp", "card_retreat_cost",
    # local retreat [30:32]
    "retreat_hp_ratio", "retreat_best_bench_hp",
    # local end [32]
    "is_end",
    # per-option features [33:37] (attack + card identity)
    "atk_effective_damage", "atk_ko_after_weakness",
    "card_is_bellibolt_ex", "card_is_voltorb",
    # combo-execution per-option features [37:43]
    "attach_target_is_voltorb",       # [37] FIXED: was dead retreat_to_kilowattrel
    "attach_target_is_bellibolt",     # [38] new
    "attach_energy_is_lightning",     # [39] new
    "attach_to_ready_attacker",       # [40] new
    "voltaic_chain_would_ko",         # [41] new: Voltorb attack KO signal
    "n_voltorb_on_bench",             # [42] new: bench swarm width
    # global [43:59] (original 16)
    "g_turn_norm", "g_my_prizes", "g_opp_prizes", "g_hand_size",
    "g_my_bench", "g_opp_bench", "g_my_active_hp", "g_my_active_maxhp",
    "g_opp_active_hp", "g_opp_active_maxhp", "g_my_energy_count",
    "g_supporter_played", "g_energy_attached", "g_retreated", "g_n_options", "g_ctx",
    # new global features [59:66]
    "g_prize_diff", "g_opp_active_hp_ratio", "g_my_bench_energy_total",
    "g_total_lightning_on_board", "g_n_bellibolt_in_play",
    "g_my_active_is_voltorb", "g_n_my_pokemon_in_play",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"{len(FEATURE_NAMES)} != {FEATURE_DIM}"
