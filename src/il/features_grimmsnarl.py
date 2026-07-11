"""
features_grimmsnarl.py — Per-option feature matrix for the Grimmsnarl IL ranker (v018).

Single-expert BC clone of kazuki0123 #3 (Marnie's Grimmsnarl ex / Munkidori toolbox,
teamId 16441533, subId 54203196, mu=1195, wr=0.67). MIDRANGE deck — 72/72 sampled
expert attacks were Grimmsnarl ex "Shadow Bullet"; this is the SAME profile as the
proven Crustle clone (v007/v009a, 906-930 live), so we mirror that recipe (generic
skeleton: option-type one-hot, damage/HP/energy/prize locals+globals) and add
Grimmsnarl-specific flags on top. Works entirely from raw obs_dict (same format as
live agent observation and replay JSON). No libcg required; all card metadata is
loaded from EN_Card_Data.csv.

Deck archetype: Grimmsnarl ex toolbox (kazuki0123's exact decklist)
  - Impidimp (646) -> Morgrem (647) -> Grimmsnarl ex (648) via Rare Candy (1079)
    on the Basic, or normal Stage1->Stage2 evolution from Morgrem in play.
  - Win-con: Grimmsnarl ex 648 attack "Shadow Bullet" (cost {D}{D}):
        180 damage to opponent's Active. Also does 30 damage to 1 of opponent's
        Benched Pokemon (no weakness/resistance applied to bench damage).
    Grimmsnarl ex only has ONE attack, so any ATTACK option while Grimmsnarl ex
    is active IS Shadow Bullet.
  - Ability "Punk Up" (648, on-evolve trigger): search deck for up to 5 Basic
    {D} Energy and attach to Marnie's Pokemon (fast energy ramp).
  - Munkidori (112) support tech: ability "Adrena-Brain" (needs {D} energy
    attached to Munkidori) moves up to 3 damage counters between Pokemon —
    occasional finisher/counter-sniping, NOT the main win-con.
  - Draw engine: Dunsparce (305) / Dudunsparce (66, ability "Run Away Draw":
    draw 3, shuffle self back), Dawn (1231), Lillie's Determination (1227).
  - Stadium: Spikemuth Gym (1259) — search deck for a Marnie's Pokemon to hand.
  - Tool: Hero's Cape (1159, ACE SPEC, +100 HP).
  - Energy: Basic {D} Energy = id 7 (x12 in deck).

Feature layout (per row = one option), total FEATURE_DIM = 72:
  [0..16]   Option type one-hot (17 types: NUMBER=0 .. SPECIAL_CONDITION=16)
  --- Local per-option features (slots 17..42, 26 slots) ---
  [17]  ATTACK: damage (base, from CSV)
  [18]  ATTACK: target current HP
  [19]  ATTACK: is-KO flag (damage >= target_hp)                 [== "would_ko"]
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
  --- Grimmsnarl-specific local [35..42] ---
  [35]  ATTACK: is_shadow_bullet_attack (my active id==648 and opt is ATTACK;
        Grimmsnarl ex has exactly 1 attack, so this is always true for its attacks)
  [36]  EVOLVE/PLAY: evolves_toward_grimmsnarl (card id in {647, 648})
  [37]  EVOLVE/PLAY: via_rare_candy (card id == 1079)
  [38]  ATTACH: attach_D_to_grimmsnarl_line (energy card id==7 {D} AND target
        pokemon id in {646, 647, 648})
  [39]  ABILITY: munkidori_adrena_brain_option (ability-source pokemon id==112,
        resolved via option area/index — 4=ACTIVE, 5=BENCH)
  [40]  PLAY/EVOLVE/CARD: card_is_munkidori (card id == 112)
  [41]  PLAY/EVOLVE/CARD: card_is_dunsparce (card id == 305)
  [42]  PLAY/EVOLVE/CARD: card_is_dudunsparce (card id == 66)
  --- Global state features (slots 43..71, 29 values) ---
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
  [53]  energy on my active (total energy count)
  [54]  supporter_played flag
  [55]  energy_attached flag
  [56]  retreated flag
  [57]  n_options in this decision
  [58]  select_context (raw int)
  [59]  prize_diff = opp_prizes_remaining - my_prizes_remaining (positive = we're ahead)
  [60]  opp_active_hp_ratio = opp_active_hp / max(1, opp_active_maxhp)
  --- Grimmsnarl-specific global [61..71] ---
  [61]  dark_energy_on_active = count of {D} Basic Energy (id==7) attached to my active
  [62]  grimmsnarl_is_active = 1 if my active pokemon id==648
  [63]  has_grimmsnarl_in_play = 1 if 648 in my active or bench
  [64]  can_evolve_to_grimmsnarl_this_turn = 1 if (646 or 647 in play) AND
        (648 in hand) AND (1079 in hand OR 647 in play)
  [65]  has_buddy_buddy_poffin_in_hand = 1 if 1086 in hand
  [66]  has_poke_pad_in_hand = 1 if 1152 in hand
  [67]  has_dawn_in_hand = 1 if 1231 in hand
  [68]  has_lillies_determination_in_hand = 1 if 1227 in hand
  [69]  has_xerosics_machinations_in_hand = 1 if 1197 in hand
  [70]  has_spikemuth_gym_in_play = 1 if 1259 in my active/bench/stadium-in-play
        (approximated: card id 1259 seen among my in-play pokemon is impossible —
        stadium is tracked separately in obs; see _has_stadium_in_play)
  [71]  has_heros_cape_in_hand = 1 if 1159 in hand

Total: 72 features per option. (Wider than the 66-dim Crustle/Alakazam modules
because the Grimmsnarl spec calls for individually-reportable flags across 11
distinct card ids — same architecture/contract, same defensive coding style,
just a few more named slots.)

Obs field access notes: see features_v011.py / features_alakazam.py docstrings
(same schema). Additionally:
  pokemon dict 'energies': list of energy CARD IDs (not EnergyType ints) attached
    (e.g. [7, 7] for two Basic {D} Energy).
  ABILITY option dict: {area, index, type=10}. area is AreaType (4=ACTIVE,
    5=BENCH, 7=STADIUM). Resolve source pokemon via (area, index) same as
    ATTACH's target resolution.
  obs['current']['stadium']: stadium card dict (id) if present, else None/absent
    (schema varies; probed defensively).
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
N_GLOBAL = 29      # slots 43-71
FEATURE_DIM = N_OPT_TYPES + N_LOCAL + N_GLOBAL  # = 72

# Grimmsnarl deck card IDs
IMPIDIMP_ID = 646
MORGREM_ID = 647
GRIMMSNARL_ID = 648
MORPEKO_ID = 649
MUNKIDORI_ID = 112
DUNSPARCE_ID = 305
DUDUNSPARCE_ID = 66
RARE_CANDY_ID = 1079
BUDDY_BUDDY_POFFIN_ID = 1086
POKE_PAD_ID = 1152
DAWN_ID = 1231
LILLIES_DETERMINATION_ID = 1227
XEROSICS_MACHINATIONS_ID = 1197
NIGHT_STRETCHER_ID = 1097
ENERGY_SEARCH_ID = 1119
ENERGY_RECYCLER_ID = 1139
HEROS_CAPE_ID = 1159
SPIKEMUTH_GYM_ID = 1259
BASIC_D_ENERGY_ID = 7

GRIMMSNARL_LINE = {IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_ID}
GRIMMSNARL_EVOLVE_TARGETS = {MORGREM_ID, GRIMMSNARL_ID}

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

# AreaType ints (mirrors cg/api.py AreaType)
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_STADIUM = 7

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

        # IMPORTANT: filter OUT ability rows before building the attack-damage
        # list. The CSV lists one row per move (attacks AND abilities) for a
        # card, in file order. In-game ATTACK options only correspond to the
        # non-ability rows, so `attack_rank` (computed from in-game ATTACK
        # option order) must index into a damages list that excludes ability
        # rows — otherwise a card whose ability row precedes its attack row
        # (e.g. Grimmsnarl ex 648: "[Ability] Punk Up" then "Shadow Bullet")
        # gets its attack damage silently read as the ability's NaN->0.
        _atk_rows = [
            r for _, r in grp.iterrows()
            if not str(r.get("Move Name") or "").strip().startswith("[Ability]")
        ]
        damages = [_parse_damage(r.get("Damage")) for r in _atk_rows]
        move_names = [str(r.get("Move Name") or "").lower() for r in _atk_rows]

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
    """
    Resolve a pokemon from the player state dict using area + index.
    area: 4=ACTIVE, 5=BENCH
    """
    if area == AREA_ACTIVE:
        active = player_dict.get("active") or []
        return active[0] if active else None
    elif area == AREA_BENCH:
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


def _has_stadium_in_play(obs_dict: dict, stadium_id: int) -> bool:
    """Best-effort detection of a stadium card in play, tolerant of schema
    variance across obs versions. Checks a few plausible field names."""
    cur = obs_dict.get("current") or {}
    for key in ("stadium", "field", "stadiumCard"):
        val = cur.get(key)
        if isinstance(val, dict) and int(val.get("id") or 0) == stadium_id:
            return True
        if isinstance(val, list):
            for v in val:
                if isinstance(v, dict) and int(v.get("id") or 0) == stadium_id:
                    return True
    return False


def build_feature_matrix(obs_dict: dict) -> np.ndarray:
    """
    Build per-option feature matrix from a raw obs_dict.

    Returns X: np.ndarray of shape (n_options, FEATURE_DIM=72), float32.
    Rows correspond to obs_dict['select']['option'] in order. Never raises;
    all field access is fully defensive.
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
    dark_energy_on_active = float(sum(
        1 for e in ((my_active or {}).get("energies") or []) if int(e) == BASIC_D_ENERGY_ID
    ))

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

    # ── Grimmsnarl combo/global signals ────────────────────────────────────────
    my_play_ids = _ids_in_play(my_pl)
    grimmsnarl_is_active = float(my_active_card_id == GRIMMSNARL_ID)
    has_grimmsnarl_in_play = float(GRIMMSNARL_ID in my_play_ids)
    has_impidimp_in_play = IMPIDIMP_ID in my_play_ids
    has_morgrem_in_play = MORGREM_ID in my_play_ids
    has_grimmsnarl_in_hand = GRIMMSNARL_ID in my_hand_ids
    n_rare_candy_in_hand = sum(1 for cid in my_hand_ids if cid == RARE_CANDY_ID)
    can_evolve_to_grimmsnarl_this_turn = float(
        (has_impidimp_in_play or has_morgrem_in_play)
        and has_grimmsnarl_in_hand
        and (n_rare_candy_in_hand > 0 or has_morgrem_in_play)
    )

    has_poffin_in_hand = float(BUDDY_BUDDY_POFFIN_ID in my_hand_ids)
    has_poke_pad_in_hand = float(POKE_PAD_ID in my_hand_ids)
    has_dawn_in_hand = float(DAWN_ID in my_hand_ids)
    has_lillies_det_in_hand = float(LILLIES_DETERMINATION_ID in my_hand_ids)
    has_xerosics_in_hand = float(XEROSICS_MACHINATIONS_ID in my_hand_ids)
    has_heros_cape_in_hand = float(HEROS_CAPE_ID in my_hand_ids)
    has_spikemuth_in_play = float(_has_stadium_in_play(obs_dict, SPIKEMUTH_GYM_ID))

    prize_diff = float(opp_prizes - my_prizes)
    opp_active_hp_ratio = float(opp_active_hp) / float(max(1, opp_active_maxhp))

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
        dark_energy_on_active,
        grimmsnarl_is_active,
        has_grimmsnarl_in_play,
        can_evolve_to_grimmsnarl_this_turn,
        has_poffin_in_hand,
        has_poke_pad_in_hand,
        has_dawn_in_hand,
        has_lillies_det_in_hand,
        has_xerosics_in_hand,
        has_spikemuth_in_play,
        has_heros_cape_in_hand,
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

            # [B+18] is_shadow_bullet_attack: Grimmsnarl ex has exactly 1 attack,
            # so any ATTACK option while it's active IS Shadow Bullet.
            row[B + 18] = float(my_active_card_id == GRIMMSNARL_ID)

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
                if area == AREA_HAND and idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[idx].get("id") or 0)
                in_area = int(opt.get("inPlayArea") or 0)
                in_idx = int(opt.get("inPlayIndex") or 0)
                target_pk = _get_pokemon_from_obs(my_pl, in_area, in_idx)

            elif opt_type == OPT_EVOLVE:
                area = int(opt.get("area") or 0)
                idx = int(opt.get("index") or 0)
                if area == AREA_HAND and idx < len(my_hand_cards):
                    card_id = int(my_hand_cards[idx].get("id") or 0)
                in_area = int(opt.get("inPlayArea") or 0)
                in_idx = int(opt.get("inPlayIndex") or 0)
                target_pk = _get_pokemon_from_obs(my_pl, in_area, in_idx)

            elif opt_type == OPT_CARD:
                area = int(opt.get("area") or 0)
                pidx = int(opt.get("playerIndex") or 0)
                idx = int(opt.get("index") or 0)
                pl = players[pidx] if pidx < len(players) else {}
                if area == AREA_HAND:
                    h = pl.get("hand") or []
                    if idx < len(h):
                        card_id = int(h[idx].get("id") or 0)
                elif area == AREA_ACTIVE:
                    act = pl.get("active") or []
                    if act:
                        card_id = int(act[0].get("id") or 0)
                elif area == AREA_BENCH:
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

            # [B+19] evolves_toward_grimmsnarl: this option's card targets 647/648
            row[B + 19] = float(card_id in GRIMMSNARL_EVOLVE_TARGETS)
            # [B+20] via_rare_candy
            row[B + 20] = float(card_id == RARE_CANDY_ID)
            # [B+21] attach_D_to_grimmsnarl_line
            if opt_type == OPT_ATTACH:
                target_id = int((target_pk or {}).get("id") or 0)
                row[B + 21] = float(card_id == BASIC_D_ENERGY_ID and target_id in GRIMMSNARL_LINE)
            # [B+23..25] specific card-id flags
            row[B + 23] = float(card_id == MUNKIDORI_ID)
            row[B + 24] = float(card_id == DUNSPARCE_ID)
            row[B + 25] = float(card_id == DUDUNSPARCE_ID)

        elif opt_type == OPT_RETREAT:
            ratio = my_active_hp / my_active_maxhp
            row[B + 13] = ratio
            bench_list = my_pl.get("bench") or []
            best_bench_hp = max((b.get("hp") or 0 for b in bench_list), default=0)
            row[B + 14] = float(best_bench_hp)

        elif opt_type == OPT_END:
            row[B + 15] = 1.0

        elif opt_type == OPT_ABILITY:
            # [B+22] munkidori_adrena_brain_option: resolve source pokemon via
            # (area, index) — ABILITY options carry {area, index, type} only.
            area = int(opt.get("area") or 0)
            idx = int(opt.get("index") or 0)
            src_pk = _get_pokemon_from_obs(my_pl, area, idx)
            src_id = int((src_pk or {}).get("id") or 0)
            row[B + 22] = float(src_id == MUNKIDORI_ID)

        # global features
        row[N_OPT_TYPES + N_LOCAL: N_OPT_TYPES + N_LOCAL + N_GLOBAL] = global_vec

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
    # Grimmsnarl-specific local [33:43]
    "atk_effective_damage", "atk_ko_after_weakness",
    "atk_is_shadow_bullet",
    "card_evolves_toward_grimmsnarl", "card_via_rare_candy",
    "attach_D_to_grimmsnarl_line",
    "ability_is_munkidori_adrena_brain",
    "card_is_munkidori", "card_is_dunsparce", "card_is_dudunsparce",
    # global [43:61]
    "g_turn_norm", "g_my_prizes", "g_opp_prizes", "g_hand_size",
    "g_my_bench", "g_opp_bench", "g_my_active_hp", "g_my_active_maxhp",
    "g_opp_active_hp", "g_opp_active_maxhp", "g_my_energy_count",
    "g_supporter_played", "g_energy_attached", "g_retreated", "g_n_options", "g_ctx",
    "g_prize_diff", "g_opp_active_hp_ratio",
    # Grimmsnarl-specific global [61:72]
    "g_dark_energy_on_active", "g_grimmsnarl_is_active", "g_has_grimmsnarl_in_play",
    "g_can_evolve_to_grimmsnarl_this_turn",
    "g_has_buddy_buddy_poffin_in_hand", "g_has_poke_pad_in_hand", "g_has_dawn_in_hand",
    "g_has_lillies_determination_in_hand", "g_has_xerosics_machinations_in_hand",
    "g_has_spikemuth_gym_in_play", "g_has_heros_cape_in_hand",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"{len(FEATURE_NAMES)} != {FEATURE_DIM}"
