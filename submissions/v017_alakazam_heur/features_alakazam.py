"""
features_alakazam.py — Per-option feature matrix for the Alakazam IL ranker (v016).

Single-expert BC clone of aidy #8 (Alakazam "Powerful Hand" combo deck).
Works entirely from raw obs_dict (same format as live agent observation and replay JSON).
No libcg required; all card metadata is loaded from EN_Card_Data.csv.

Deck archetype: Alakazam combo
  - Abra (741) → Kadabra (742) → Alakazam (743) via Rare Candy (1079) on T3-T4
  - Win-con: Alakazam 743 attack "Powerful Hand" (cost {P}, attackId 1072):
        Place 2 damage counters on opp Active PER CARD IN YOUR HAND
        = 20 × handN effective damage (damage COUNTERS bypass "prevent damage" shields;
          Alakazam is non-ex → double bypass).
        OHKOs at handN ~13-17.
  - Hand-ramp: Kadabra +2 on evo, Alakazam +3 on evo (Psychic Draw ability),
    Dudunsparce (66) +3, Enriching Energy (13) +4, Hilda (1225)/Dawn (1231)/Poké Pad (1152).
  - Stadium Battle Cage (1264) protects bench.
  - HARD COUNTER: Cubchoo (506) "Snotted Up" locks attack USE → blocks Powerful Hand.

Feature layout (per row = one option), total FEATURE_DIM = 66:
  [0..16]   Option type one-hot (17 types: NUMBER=0 .. SPECIAL_CONDITION=16)
  --- Local per-option features (slots 17..42, 26 slots) ---
  [17]  ATTACK: damage (base, from CSV)
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
  [35]  ATTACK: is_powerful_hand (active card id==743 and opt type==ATTACK)
  [36]  ATTACK: powerful_hand_est_dmg_norm (min(20*hand_size,300)/300)  [combo damage estimate]
  [37]  ATTACK: powerful_hand_would_ko (is_powerful_hand AND 20*hand_size >= opp_active_hp)
  [38]  ATTACH: target_is_alakazam_or_line (target pokemon id in {743,742,741})
  [39]  ATTACH: energy_is_psychic (energy card id in {5,19,13} — all provide {P})
  [40]  EVOLVE/PLAY: evolves_to_alakazam or card_is_alakazam (card id==743)
  [41]  EVOLVE/PLAY: via_rare_candy (card id==1079)
  [42]  hand_size_delta_if_chosen (estimate: PLAY/ATTACH ≈ -1, ATTACK/RETREAT/END ≈ 0;
        encodes "don't dump hand before attacking with Alakazam" — core combo-preservation)
  --- Global state features (slots 43..65, 23 values) ---
  [43]  turn / 100 (normalized)
  [44]  my prizes left
  [45]  opp prizes left
  [46]  my hand size  (THE combo driver — Powerful Hand dmg = 20×this)
  [47]  my bench count
  [48]  opp bench count
  [49]  my active HP
  [50]  my active maxHP
  [51]  opp active HP
  [52]  opp active maxHP
  [53]  energy on my active (total energy count)
  [54]  supporter_played flag
  [55]  energy_attached flag
  [56]  retreated flag
  [57]  n_options in this decision
  [58]  select_context (raw int)
  [59]  prize_diff = opp_prizes_remaining - my_prizes_remaining  (positive = we're ahead)
  [60]  opp_active_hp_ratio = opp_active_hp / max(1, opp_active_maxhp)
  [61]  alakazam_is_active = 1 if my active pokemon id==743
  [62]  has_alakazam_in_play = 1 if 743 in my active or bench
  [63]  has_kadabra_in_play = 1 if 742 in my active or bench
  [64]  n_rare_candy_in_hand = count of id 1079 in hand, /4
  [65]  can_evolve_to_alakazam_this_turn = 1 if (741 or 742 in play) AND (743 in hand) AND
        (1079 in hand OR 742 in play [normal Stage1→Stage2 evo])

Total: 66 features per option.
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
N_LOCAL = 26       # slots 17-42
N_GLOBAL = 23      # slots 43-65
FEATURE_DIM = N_OPT_TYPES + N_LOCAL + N_GLOBAL  # = 66

# Alakazam line card IDs
ABRA_ID = 741
KADABRA_ID = 742
ALAKAZAM_ID = 743
RARE_CANDY_ID = 1079
CUBCHOO_ID = 506
DUNSPARCE_ID = 305
DUDUNSPARCE_ID = 66
BATTLE_CAGE_ID = 1264
POKEMON_BREEDER_ID = 1152   # Poké Pad
HILDA_ID = 1225
DAWN_ID = 1231
POFFIN_ID = 1086
BOSS_ID = 1182
NIGHT_STRETCHER_ID = 1097
ENHANCED_HAMMER_ID = 1081
SACRED_ASH_ID = 1129
LANA_ID = 1184
ENRICHING_ENERGY_ID = 13
BASIC_PSYCHIC_ENERGY_ID = 5
TELEPATH_PSYCHIC_ENERGY_ID = 19

# Psychic-providing energy card ids
PSYCHIC_ENERGY_IDS = {BASIC_PSYCHIC_ENERGY_ID, TELEPATH_PSYCHIC_ENERGY_ID, ENRICHING_ENERGY_ID}

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
PSYCHIC_ENERGY_TYPE = 5


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
        _base / "EN_Card_Data.csv",
        _base.parent.parent / "data" / "EN_Card_Data.csv",
        _base.parent.parent.parent / "data" / "EN_Card_Data.csv",
        Path("/kaggle_simulations/agent/EN_Card_Data.csv"),
        Path("EN_Card_Data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


_DATA_PATH = _find_card_data_csv()


def _parse_energy_str(s: str) -> int | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    return _ENERGY_STR_MAP.get(s)


def _parse_damage(s: Any) -> int:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0
    try:
        return int(re.sub(r"[^0-9]", "", str(s)) or "0")
    except Exception:
        return 0


@lru_cache(maxsize=1)
def _load_card_db() -> dict[int, dict]:
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

        damages = [_parse_damage(r.get("Damage")) for _, r in grp.iterrows()]

        # attack names (lowercased) — to identify Powerful Hand by name as a fallback
        move_names = [str(r.get("Move Name") or "").lower() for _, r in grp.iterrows()]

        card_info[int(card_id)] = {
            "hp": hp,
            "retreat": retreat,
            "is_pokemon": is_pokemon,
            "is_supporter": is_supporter,
            "is_item": is_item,
            "is_tool": is_tool,
            "is_energy": is_energy,
            "is_stadium": is_stadium,
            "card_type_int": ct,
            "weakness_energy": weak_e,
            "own_energy": own_energy,
            "is_ex": is_ex,
            "is_mega_ex": is_mega_ex,
            "damages": damages,
            "move_names": move_names,
        }

    return card_info


def _get_card(card_id: int) -> dict:
    db = _load_card_db()
    return db.get(int(card_id) if card_id else 0, {})


def _get_pokemon_from_obs(player_dict: dict, area: int, idx: int) -> dict | None:
    if area == 4:
        active = player_dict.get("active") or []
        return active[0] if active else None
    elif area == 5:
        bench = player_dict.get("bench") or []
        return bench[idx] if idx < len(bench) else None
    return None


def _ids_in_play(player_dict: dict) -> set[int]:
    """Card ids of all my pokemon in play (active + bench)."""
    ids = set()
    for pk in (player_dict.get("active") or []):
        if isinstance(pk, dict) and pk.get("id") is not None:
            ids.add(int(pk["id"]))
    for pk in (player_dict.get("bench") or []):
        if isinstance(pk, dict) and pk.get("id") is not None:
            ids.add(int(pk["id"]))
    return ids


def build_feature_matrix(obs_dict: dict) -> np.ndarray:
    """Build per-option feature matrix from a raw obs_dict.

    Returns X: np.ndarray of shape (n_options, FEATURE_DIM=66), float32.
    Rows correspond to obs_dict['select']['option'] in order. Never raises.
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

    my_active_card_id = int((my_active or {}).get("id") or 0)
    my_card = _get_card(my_active_card_id)
    my_own_energy = my_card.get("own_energy")

    opp_active_card_id = int((opp_active or {}).get("id") or 0)
    opp_card = _get_card(opp_active_card_id)
    opp_weakness_energy = opp_card.get("weakness_energy")
    opp_prize_val = 3.0 if opp_card.get("is_mega_ex") else (2.0 if opp_card.get("is_ex") else 1.0)

    my_hand_cards = my_pl.get("hand") or []
    my_hand_ids = [int((c or {}).get("id") or 0) for c in my_hand_cards]

    # ── Alakazam combo global signals ─────────────────────────────────────────
    my_play_ids = _ids_in_play(my_pl)
    alakazam_is_active = float(my_active_card_id == ALAKAZAM_ID)
    has_alakazam_in_play = float(ALAKAZAM_ID in my_play_ids)
    has_kadabra_in_play = float(KADABRA_ID in my_play_ids)
    has_abra_in_play = float(ABRA_ID in my_play_ids)
    n_rare_candy_in_hand = sum(1 for cid in my_hand_ids if cid == RARE_CANDY_ID)
    has_alakazam_in_hand = any(cid == ALAKAZAM_ID for cid in my_hand_ids)
    can_evolve_to_alakazam = float(
        (has_abra_in_play or has_kadabra_in_play)
        and has_alakazam_in_hand
        and (n_rare_candy_in_hand > 0 or has_kadabra_in_play)
    )

    prize_diff = float(opp_prizes - my_prizes)
    opp_active_hp_ratio = float(opp_active_hp) / float(max(1, opp_active_maxhp))

    # powerful-hand effective damage estimate (2 damage counters per card = 20*dmg per card)
    powerful_hand_dmg = 20 * my_hand_count

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
        prize_diff,
        opp_active_hp_ratio,
        alakazam_is_active,
        has_alakazam_in_play,
        has_kadabra_in_play,
        float(n_rare_candy_in_hand) / 4.0,
        can_evolve_to_alakazam,
    ], dtype=np.float32)
    assert len(global_vec) == N_GLOBAL

    X = np.zeros((n_opts, FEATURE_DIM), dtype=np.float32)

    for i, opt in enumerate(opts):
        row = X[i]
        opt_type = int(opt.get("type") or 0)

        if 0 <= opt_type < N_OPT_TYPES:
            row[opt_type] = 1.0

        B = N_OPT_TYPES  # = 17

        if opt_type == OPT_ATTACK:
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

            eff_damage = float(damage * 2 if wk_match else damage)
            row[B + 16] = eff_damage
            row[B + 17] = float(eff_damage > 0 and eff_damage >= opp_active_hp)

            # Alakazam Powerful Hand detection:
            # active is 743 and this is an ATTACK option → only attack is Powerful Hand.
            # (Falls back to attack-name match if active id unknown.)
            is_ph = False
            if my_active_card_id == ALAKAZAM_ID:
                is_ph = True
            else:
                move_names = my_card.get("move_names") or []
                if attack_rank < len(move_names) and "powerful hand" in move_names[attack_rank]:
                    is_ph = True
            row[B + 18] = float(is_ph)
            row[B + 19] = float(min(powerful_hand_dmg, 300) / 300.0) if is_ph else 0.0
            row[B + 20] = float(is_ph and powerful_hand_dmg >= opp_active_hp and opp_active_hp > 0)

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
                if area == 2 and idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[idx].get("id") or 0)
                in_area = int(opt.get("inPlayArea") or 0)
                in_idx = int(opt.get("inPlayIndex") or 0)
                target_pk = _get_pokemon_from_obs(my_pl, in_area, in_idx)

            elif opt_type == OPT_EVOLVE:
                area = int(opt.get("area") or 0)
                idx = int(opt.get("index") or 0)
                if area == 2 and idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[idx].get("id") or 0)
                # evolution target pokemon
                in_area = int(opt.get("inPlayArea") or 0)
                in_idx = int(opt.get("inPlayIndex") or 0)
                target_pk = _get_pokemon_from_obs(my_pl, in_area, in_idx)

            elif opt_type == OPT_CARD:
                area = int(opt.get("area") or 0)
                pidx = int(opt.get("playerIndex") or 0)
                idx = int(opt.get("index") or 0)
                pl = players[pidx] if pidx < len(players) else {}
                if area == 2:
                    h = pl.get("hand") or []
                    if idx < len(h):
                        card_id = int(h[idx].get("id") or 0)
                elif area == 4:
                    act = pl.get("active") or []
                    if act:
                        card_id = int(act[0].get("id") or 0)
                elif area == 5:
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

            # Alakazam-specific local features
            # [B+21] attach target is Alakazam line (743/742/741)
            if opt_type == OPT_ATTACH and target_pk is not None:
                tid = int(target_pk.get("id") or 0)
                row[B + 21] = float(tid in (ALAKAZAM_ID, KADABRA_ID, ABRA_ID))
                # [B+22] energy is psychic (card id in {5,19,13})
                row[B + 22] = float(card_id in PSYCHIC_ENERGY_IDS)
            # [B+23] evolves_to_alakazam / card is alakazam (743)
            row[B + 23] = float(card_id == ALAKAZAM_ID)
            # [B+24] via_rare_candy (1079)
            row[B + 24] = float(card_id == RARE_CANDY_ID)

        elif opt_type == OPT_RETREAT:
            ratio = my_active_hp / my_active_maxhp
            row[B + 13] = ratio
            bench_list = my_pl.get("bench") or []
            best_bench_hp = max((b.get("hp") or 0 for b in bench_list), default=0)
            row[B + 14] = float(best_bench_hp)

        elif opt_type == OPT_END:
            row[B + 15] = 1.0

        # [B+25] hand_size_delta_if_chosen (combo-preservation signal)
        # Estimate: PLAY/ATTACH/EVOLVE-from-hand ≈ -1 (consumes a hand card);
        # ATTACK/RETREAT/END/TO_BENCH ≈ 0; TO_HAND ≈ +1.
        if opt_type in (OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_CARD, OPT_ENERGY_CARD, OPT_TOOL_CARD):
            row[B + 25] = -1.0
        elif opt_type in (OPT_ATTACK, OPT_RETREAT, OPT_END, OPT_ABILITY, OPT_DISCARD):
            row[B + 25] = 0.0
        else:
            row[B + 25] = 0.0

        # global features
        row[N_OPT_TYPES + N_LOCAL : N_OPT_TYPES + N_LOCAL + N_GLOBAL] = global_vec

    return X


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
    # Alakazam-specific local [33:43]
    "atk_effective_damage", "atk_ko_after_weakness",
    "atk_is_powerful_hand", "atk_powerful_hand_est_dmg_norm", "atk_powerful_hand_would_ko",
    "attach_target_is_alakazam_line", "attach_energy_is_psychic",
    "card_is_alakazam", "card_is_rare_candy",
    "hand_size_delta_if_chosen",
    # global [43:66]
    "g_turn_norm", "g_my_prizes", "g_opp_prizes", "g_hand_size",
    "g_my_bench", "g_opp_bench", "g_my_active_hp", "g_my_active_maxhp",
    "g_opp_active_hp", "g_opp_active_maxhp", "g_my_energy_count",
    "g_supporter_played", "g_energy_attached", "g_retreated", "g_n_options", "g_ctx",
    "g_prize_diff", "g_opp_active_hp_ratio",
    "g_alakazam_is_active", "g_has_alakazam_in_play", "g_has_kadabra_in_play",
    "g_n_rare_candy_in_hand", "g_can_evolve_to_alakazam",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"{len(FEATURE_NAMES)} != {FEATURE_DIM}"