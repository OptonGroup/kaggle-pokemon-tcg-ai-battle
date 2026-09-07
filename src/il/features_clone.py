"""
features_clone.py — Per-option feature matrix for the IL ranker (Crustle+MegaKangaskhan clone).

Adapted from features_v009a.py (pure-Crustle deck) for the top-10 player's
Crustle+MegaKangaskhan ex deck (懒惰的金枪鱼 / __Taichicchi__), where Mega
Kangaskhan ex (id 756, HP300, Colorless) is the MAIN attacker and distinctive
trainer decisions (hand-disruption, heal, gust, Ace Spec tool, stadium) drive
the win. The Crustle-specific feature slots from v009a are replaced with
Kangaskhan-deck-specific ones so the model can learn the top player's strategy.

Works entirely from raw obs_dict (same format as live agent observation and
replay JSON). No libcg required; all card metadata is loaded from
data/EN_Card_Data.csv (or co-located EN_Card_Data.csv in submission/helper dir).

Feature layout (per row = one option), total FEATURE_DIM = 75:
  [0..16]   Option type one-hot (17 types: NUMBER=0 .. SPECIAL_CONDITION=16)
  --- Local per-option features (slots 17..46, 30 slots) ---
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
  --- Deck-specific LOCAL per-option features (slots 35..44, 12 slots) ---
  [35]  PLAY/EVOLVE/CARD: is_kangaskhan        (card id == 756)
  [36]  PLAY/EVOLVE/CARD: is_crustle           (card id == 345)
  [37]  PLAY/EVOLVE/CARD: is_dwebble           (card id == 344)
  [38]  PLAY/EVOLVE/CARD: is_hand_trimmer      (card id == 1087)
  [39]  PLAY/EVOLVE/CARD: is_heal_card         (card id == 1147, Jumbo Ice Cream)
  [40]  PLAY/EVOLVE/CARD: is_heros_cape        (card id == 1159, Ace Spec +HP tool)
  [41]  PLAY/EVOLVE/CARD: is_stadium           (card id == 1264, Battle Cage)
  [42]  PLAY/EVOLVE/CARD: is_boss_orders       (card id == 1182, gust)
  [43]  PLAY/EVOLVE/CARD: is_search_card       (card id in {1086, 1122}, Poffin/Pokégear)
  [44]  PLAY/EVOLVE/CARD: is_switch            (card id == 1123)
  [45]  RETREAT: retreat_to_kangaskhan         (bench target id == 756)
  [46]  RETREAT: retreat_to_crustle            (bench target id == 345)
  --- Global state features (slots 47..74, 28 values) ---
  [47]  turn / 100 (normalized)
  [48]  my prizes left
  [49]  opp prizes left
  [50]  my hand size
  [51]  my bench count
  [52]  opp bench count
  [53]  my active HP
  [54]  my active maxHP
  [55]  opp active HP
  [56]  opp active maxHP
  [57]  energy on my active
  [58]  supporter_played flag
  [59]  energy_attached flag
  [60]  retreated flag
  [61]  n_options in this decision
  [62]  select_context (raw int)
  [63]  prize_diff = opp_prizes_remaining - my_prizes_remaining  (positive = we're ahead)
  [64]  opp_active_hp_ratio = opp_active_hp / max(1, opp_active_maxhp)
  [65]  my_bench_energy_total = sum of energies count across my bench pokemon
  --- Deck-specific GLOBAL features (slots 66..74, 9 values) ---
  [66]  has_kangaskhan_on_bench   (any bench Pokémon id == 756)
  [67]  has_crustle_on_bench      (any bench id == 345)
  [68]  has_dwebble_in_hand       (any hand card id == 344)
  [69]  my_active_is_kangaskhan   (my active Pokémon id == 756)
  [70]  my_active_is_crustle      (my active id == 345)
  [71]  has_hand_trimmer_in_hand  (any hand card id == 1087)
  [72]  n_kangaskhan_in_hand      (count of 756 in hand)
  [73]  n_crustle_in_hand         (count of 345 in hand)
  [74]  my_active_hp_ratio        (active HP / maxHP)

Total: 75 features per option.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

N_OPT_TYPES = 17   # slots 0-16
N_LOCAL = 30       # slots 17-46  (was 21 in v009a; +9 deck-specific local)
N_GLOBAL = 28      # slots 47-74  (was 23 in v009a; +5 deck-specific global)
FEATURE_DIM = N_OPT_TYPES + N_LOCAL + N_GLOBAL  # = 75

# Deck card IDs — Crustle+MegaKangaskhan ex (懒惰的金枪鱼 / __Taichicchi__)
DWEBBLE_ID = 344
CRUSTLE_ID = 345
KANGASKHAN_ID = 756          # Mega Kangaskhan ex — main attacker
HAND_TRIMMER_ID = 1087       # hand-disruption
HEAL_CARD_ID = 1147          # Jumbo Ice Cream
HEROS_CAPE_ID = 1159         # Ace Spec +HP tool
BOSS_ORDERS_ID = 1182        # gust
POFFIN_ID = 1086             # Buddy-Buddy Poffin (search Basic)
POKEGEAR_ID = 1122           # Pokégear 3.0 (search/draw)
SWITCH_ID = 1123             # Switch
STADIUM_ID = 1264            # Battle Cage
SEARCH_CARD_IDS = {POFFIN_ID, POKEGEAR_ID}

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

def _find_card_data_csv() -> Path:
    """Find EN_Card_Data.csv relative to this file or fallback locations.

    Kaggle loads agents via exec() which does NOT set __file__, so we guard
    __file__ access and fall back to /kaggle_simulations/agent or CWD.
    """
    _kaggle = "/kaggle_simulations/agent"
    if os.path.isdir(_kaggle):
        _base = Path(_kaggle)
    else:
        try:
            _base = Path(__file__).parent
        except NameError:
            _base = Path(os.getcwd())

    candidates = [
        _base / "EN_Card_Data.csv",                                        # submission / Kaggle dir
        _base.parent.parent.parent / "data" / "EN_Card_Data.csv",          # repo root during dev
        Path("/kaggle_simulations/agent/EN_Card_Data.csv"),                 # Kaggle agent dir
        Path("EN_Card_Data.csv"),                                           # CWD
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # will fail gracefully in _load_card_db

_DATA_PATH = _find_card_data_csv()


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
        X: np.ndarray of shape (n_options, FEATURE_DIM=75), float32.
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

    # Hand cards (for PLAY / ATTACH index lookups + deck-specific hand features)
    my_hand_cards = my_pl.get("hand") or []

    # ── Global features ─────────────────────────────────────────────────────────
    prize_diff = float(opp_prizes - my_prizes)
    opp_active_hp_ratio = float(opp_active_hp) / float(max(1, opp_active_maxhp))
    my_bench_energy_total = float(sum(
        len((b or {}).get("energies") or []) for b in my_bench_list
    ))

    # Deck-specific global: bench/hand/active identity
    has_kangaskhan_on_bench = float(any(
        int((b or {}).get("id") or 0) == KANGASKHAN_ID for b in my_bench_list
    ))
    has_crustle_on_bench = float(any(
        int((b or {}).get("id") or 0) == CRUSTLE_ID for b in my_bench_list
    ))
    has_dwebble_in_hand = float(any(
        int((c or {}).get("id") or 0) == DWEBBLE_ID for c in my_hand_cards
    ))
    my_active_is_kangaskhan = float(my_active_card_id == KANGASKHAN_ID)
    my_active_is_crustle = float(my_active_card_id == CRUSTLE_ID)
    has_hand_trimmer_in_hand = float(any(
        int((c or {}).get("id") or 0) == HAND_TRIMMER_ID for c in my_hand_cards
    ))
    n_kangaskhan_in_hand = float(sum(
        1 for c in my_hand_cards if int((c or {}).get("id") or 0) == KANGASKHAN_ID
    ))
    n_crustle_in_hand = float(sum(
        1 for c in my_hand_cards if int((c or {}).get("id") or 0) == CRUSTLE_ID
    ))
    my_active_hp_ratio = float(my_active_hp) / float(max(1, my_active_maxhp))

    # Global vector (28 values = 19 generic + 9 deck-specific)
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
        # new generic global features:
        prize_diff,
        opp_active_hp_ratio,
        my_bench_energy_total,
        # deck-specific global features:
        has_kangaskhan_on_bench,
        has_crustle_on_bench,
        has_dwebble_in_hand,
        my_active_is_kangaskhan,
        my_active_is_crustle,
        has_hand_trimmer_in_hand,
        n_kangaskhan_in_hand,
        n_crustle_in_hand,
        my_active_hp_ratio,
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

            # per-option attack features (slots B+16, B+17)
            eff_damage = float(damage * 2 if wk_match else damage)
            row[B + 16] = eff_damage
            row[B + 17] = float(eff_damage > 0 and eff_damage >= opp_active_hp)

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

            # Deck-specific per-option card identity features (slots B+18..B+27)
            row[B + 18] = float(card_id == KANGASKHAN_ID)
            row[B + 19] = float(card_id == CRUSTLE_ID)
            row[B + 20] = float(card_id == DWEBBLE_ID)
            row[B + 21] = float(card_id == HAND_TRIMMER_ID)
            row[B + 22] = float(card_id == HEAL_CARD_ID)
            row[B + 23] = float(card_id == HEROS_CAPE_ID)
            row[B + 24] = float(card_id == STADIUM_ID)
            row[B + 25] = float(card_id == BOSS_ORDERS_ID)
            row[B + 26] = float(card_id in SEARCH_CARD_IDS)
            row[B + 27] = float(card_id == SWITCH_ID)

        elif opt_type == OPT_RETREAT:
            ratio = my_active_hp / my_active_maxhp
            row[B + 13] = ratio
            bench_list = my_pl.get("bench") or []
            best_bench_hp = max((b.get("hp") or 0 for b in bench_list), default=0)
            row[B + 14] = float(best_bench_hp)

            # Deck-specific per-option retreat target features (slots B+28, B+29)
            in_area = int(opt.get("inPlayArea") or 0)
            in_idx = int(opt.get("inPlayIndex") or 0)
            bench_target = _get_pokemon_from_obs(my_pl, in_area, in_idx)
            if bench_target is None:
                # Some retreat options use area/index instead of inPlayArea/inPlayIndex
                alt_area = int(opt.get("area") or 0)
                alt_idx = int(opt.get("index") or 0)
                bench_target = _get_pokemon_from_obs(my_pl, alt_area, alt_idx)
            bt_id = int((bench_target or {}).get("id") or 0)
            row[B + 28] = float(bt_id == KANGASKHAN_ID)
            row[B + 29] = float(bt_id == CRUSTLE_ID)

        elif opt_type == OPT_END:
            row[B + 15] = 1.0

        # [47:75] global features
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
    # local per-option attack features [33:35]
    "atk_effective_damage", "atk_ko_after_weakness",
    # deck-specific local per-option card identity [35:45]
    "card_is_kangaskhan", "card_is_crustle", "card_is_dwebble",
    "card_is_hand_trimmer", "card_is_heal_card", "card_is_heros_cape",
    "card_is_stadium", "card_is_boss_orders", "card_is_search_card", "card_is_switch",
    # deck-specific local per-option retreat target [45:47]
    "retreat_to_kangaskhan", "retreat_to_crustle",
    # global generic [47:66] (original 16 + 3 new generic)
    "g_turn_norm", "g_my_prizes", "g_opp_prizes", "g_hand_size",
    "g_my_bench", "g_opp_bench", "g_my_active_hp", "g_my_active_maxhp",
    "g_opp_active_hp", "g_opp_active_maxhp", "g_my_energy_count",
    "g_supporter_played", "g_energy_attached", "g_retreated", "g_n_options", "g_ctx",
    "g_prize_diff", "g_opp_active_hp_ratio", "g_my_bench_energy_total",
    # deck-specific global [66:75]
    "g_has_kangaskhan_on_bench", "g_has_crustle_on_bench", "g_has_dwebble_in_hand",
    "g_my_active_is_kangaskhan", "g_my_active_is_crustle",
    "g_has_hand_trimmer_in_hand", "g_n_kangaskhan_in_hand", "g_n_crustle_in_hand",
    "g_my_active_hp_ratio",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"{len(FEATURE_NAMES)} != {FEATURE_DIM}"