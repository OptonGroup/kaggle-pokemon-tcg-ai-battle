"""
v017_alakazam_heur — HAND-WRITTEN Alakazam "Powerful Hand" combo heuristic pilot.

Diagnosis from v016 (pure BC/IL ranker): the clone WHIFFS THE SETUP — in 6/8 losses
to random the active was never Alakazam (stuck on Abra/Kadabra/Dunsparce); lethal-KO
guard fired only 1.4%. BC can't reproduce the multi-step setup→closure plan.

Fix: a hand-written heuristic that directly encodes the setup→closure plan as the
PRIMARY policy, with the IL ranker only as a tiebreak (rule 9) for non-critical
decisions. Alakazam's win-con is a SINGLE attack with a COMPUTABLE lethal
(20 × hand_size ≥ opp HP), so the heuristic is tractable.

Win-con: Alakazam 743 attack "Powerful Hand" (cost {P}) = place 2 damage counters
on opp Active PER CARD IN YOUR HAND = 20 × my_hand_size effective damage → OHKO at
hand ~13-17. Damage COUNTERS bypass "prevent damage done by attacks" shields.

Heuristic policy (priority order, first match wins each decision):
  0. Deck select (obs.select is None) → return the 60-id list.
  1. Never-throw wrapper (try/except → first-legal-minCount fallback).
  2. Lethal closure (TOP): if my active is Alakazam 743 AND 20×my_hand_size ≥
     opp_active_remaining_hp AND an ATTACK option is offered → choose it.
  3. Setup — evolve toward Alakazam: EVOLVE option whose hand card is 743 or 1079
     (Rare Candy → Alakazam) first (prefer targeting the ACTIVE so Alakazam lands
     active); else EVOLVE with hand card 742 (Abra→Kadabra, step toward Alakazam).
  4. Energy attach to Alakazam: if an Alakazam 743 in play lacks a Psychic energy,
     take an ATTACH option that attaches a Psychic energy (id 5 or 19) TO Alakazam.
  5. Get Alakazam active: if Alakazam is on bench, active is NOT Alakazam, and no
     evolve-to-Alakazam option is available this turn → take a RETREAT option; on
     the follow-up TO_ACTIVE/SWITCH prompt, pick the bench Alakazam.
  6. Hand-ramp before attacking: if Alakazam is active + has Psychic energy + NOT
     lethal yet (20×hand < opp HP) AND a hand-GROW option exists (PLAY Hilda 1225
     / Dawn 1231 / Poké Pad 1152; ABILITY Alakazam 743 or Kadabra 742 Psychic Draw
     / Dudunsparce 66 Run Away Draw; ATTACH Enriching Energy 13) → play it.
  7. Cubchoo-lock pivot: if opp active is 506 Cubchoo → take a PLAY Boss's Orders
     1182 option (gust a non-Cubchoo); on the target prompt pick highest-HP non-506.
  8. Attack fallback: if active is Alakazam + has Psychic + ATTACK offered → take
     Powerful Hand (non-lethal chip is fine, takes a prize). Else if ATTACK offered
     and no setup possible → attack with whatever active can do (don't deck out).
  9. IL tiebreak (ONLY for uncovered decisions): score the currently-legal options
     with il_alakazam.txt + features_alakazam.py and pick the top. BC only on
     off-critical-path decisions where whiffing doesn't kill the game.

Kaggle loads main.py via exec() WITHOUT __file__ → guarded _DIR resolution.
"""
from __future__ import annotations

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Kaggle exec() does NOT set __file__. Resolution order:
#   1. /kaggle_simulations/agent  (Kaggle runtime)
#   2. __file__-based dir          (local / importlib)
#   3. os.getcwd()                 (last resort)
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

# ── Constants ────────────────────────────────────────────────────────────────
ABRA_ID = 741
KADABRA_ID = 742
ALAKAZAM_ID = 743
DUNSPARCE_ID = 305
DUDUNSPARCE_ID = 66
CUBCHOO_ID = 506
RARE_CANDY_ID = 1079
POFFIN_ID = 1086
POKE_PAD_ID = 1152
ENHANCED_HAMMER_ID = 1081
NIGHT_STRETCHER_ID = 1097
SACRED_ASH_ID = 1129
BOSS_ID = 1182
DAWN_ID = 1231
HILDA_ID = 1225
LANA_ID = 1184
BATTLE_CAGE_ID = 1264
ENRICHING_ENERGY_ID = 13
BASIC_PSYCHIC_ENERGY_ID = 5
TELEPATH_PSYCHIC_ENERGY_ID = 19
PSYCHIC_ENERGY_IDS = {BASIC_PSYCHIC_ENERGY_ID, TELEPATH_PSYCHIC_ENERGY_ID}
ALL_PSYCHIC_ENERGY_IDS = {BASIC_PSYCHIC_ENERGY_ID, TELEPATH_PSYCHIC_ENERGY_ID, ENRICHING_ENERGY_ID}
HAND_GROW_SUPPORTERS = {HILDA_ID, DAWN_ID}
HAND_GROW_ITEMS = {POKE_PAD_ID}

POWERFUL_HAND_DMG_PER_CARD = 20  # 2 damage counters (=20 dmg) per card in hand

# OptionType ints (mirrors cg/api.py OptionType; matches features_alakazam.py)
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

# SelectContext values
CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_HAND = 7

PSYCHIC_ENERGY_TYPE = 5  # EnergyType.PSYCHIC

# ── Instrumentation counters (gate_runner compatible) ────────────────────────
_n_model_decisions: int = 0       # heuristic (rules 2-8) decisions
_n_ko_guard: int = 0              # rule 2 lethal closure
_n_heuristic_fallback: int = 0    # rule 9 IL tiebreak
_n_trivial_fallback: int = 0
_exception_count: int = 0
# per-rule fire counters (diagnostic)
_rule_fires: dict[int, int] = {i: 0 for i in range(2, 10)}


# ── File helpers ─────────────────────────────────────────────────────────────
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


# ── Load deck ────────────────────────────────────────────────────────────────
_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv = [line.strip() for line in _f if line.strip()]
my_deck: list[int] = [int(x) for x in _csv[:60]]
assert len(my_deck) == 60, f"deck.csv has {len(my_deck)} cards (expected 60)"

# ── Load LightGBM model + features (for rule 9 IL tiebreak) ──────────────────
try:
    import numpy as np  # noqa: E402
    import lightgbm as lgb  # noqa: E402
    _model_path = _find_file("il_alakazam.txt")
    _ranker = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[v017] WARNING: LightGBM model load failed: {_load_err}", file=sys.stderr)

try:
    from features_alakazam import build_feature_matrix  # noqa: E402
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[v017] WARNING: features_alakazam.py import failed: {_feat_err}", file=sys.stderr)


# ── obs_dict parsing helpers (raw dict; no libcg needed) ─────────────────────
def _safe_int(v, d=0) -> int:
    try:
        if v is None:
            return d
        return int(v)
    except Exception:
        return d


def _parse_state(obs_dict: dict):
    """Returns (cur, me, my, opp, my_active, opp_active, my_bench_list, hand_list)."""
    cur = obs_dict.get("current") or {}
    me = _safe_int(cur.get("yourIndex"))
    players = cur.get("players") or [{}, {}]
    while len(players) < 2:
        players = players + [{}]
    my = players[me] or {}
    opp = players[1 - me] or {}
    my_active = (my.get("active") or [None])
    my_active = my_active[0] if my_active else None
    opp_active = (opp.get("active") or [None])
    opp_active = opp_active[0] if opp_active else None
    my_bench = my.get("bench") or []
    hand_list = my.get("hand") or []
    return cur, me, my, opp, my_active, opp_active, my_bench, hand_list


def _pk_id(pk) -> int:
    if pk is None or not isinstance(pk, dict):
        return 0
    return _safe_int(pk.get("id"))


def _pk_has_psychic(pk) -> bool:
    if pk is None or not isinstance(pk, dict):
        return False
    for e in (pk.get("energies") or []):
        try:
            if int(e) == PSYCHIC_ENERGY_TYPE:
                return True
        except Exception:
            pass
    return False


def _hand_card_id(hand_list, idx) -> int:
    if idx < 0 or idx >= len(hand_list):
        return 0
    c = hand_list[idx]
    return _safe_int((c or {}).get("id")) if isinstance(c, dict) else 0


def _target_pk(my_active, my_bench, in_play_area, in_play_index):
    """inPlayArea: 4=active, 5=bench. Returns the pokemon dict or None."""
    if in_play_area == 4:
        return my_active
    elif in_play_area == 5:
        if 0 <= in_play_index < len(my_bench):
            return my_bench[in_play_index]
    return None


def _ability_pk(my_active, my_bench, opt):
    """For ABILITY option: area 4=active, 5=bench with index."""
    area = _safe_int(opt.get("area"))
    idx = _safe_int(opt.get("index"))
    if area == 4:
        return my_active
    elif area == 5:
        if 0 <= idx < len(my_bench):
            return my_bench[idx]
    return None


def _alakazam_in_play(my_active, my_bench) -> bool:
    if _pk_id(my_active) == ALAKAZAM_ID:
        return True
    return any(_pk_id(b) == ALAKAZAM_ID for b in my_bench if isinstance(b, dict))


def _bench_alakazam_index(my_bench) -> int:
    for i, b in enumerate(my_bench):
        if isinstance(b, dict) and _safe_int(b.get("id")) == ALAKAZAM_ID:
            return i
    return -1


# ── Heuristic core (rules 2-8) ───────────────────────────────────────────────
def _heuristic(obs_dict: dict, opts: list, sel: dict) -> list[int] | None:
    cur, me, my, opp, my_active, opp_active, my_bench, hand_list = _parse_state(obs_dict)
    ctx = _safe_int(sel.get("context"))
    my_active_id = _pk_id(my_active)
    opp_active_id = _pk_id(opp_active)
    opp_active_hp = _safe_int((opp_active or {}).get("hp")) if opp_active else 0
    hand_count = _safe_int(my.get("handCount"), len(hand_list))
    alakazam_active = (my_active_id == ALAKAZAM_ID)
    alakazam_in_play = _alakazam_in_play(my_active, my_bench)

    # ── Setup-phase preferences (ctx 1/2): prefer Abra as starter/bench ──────
    if ctx == CTX_SETUP_ACTIVE:
        # CARD options point to hand cards (area==2). Pick Abra 741 if offered.
        best, best_id = -1, -1
        for i, o in enumerate(opts):
            if _safe_int(o.get("type")) != OPT_CARD:
                continue
            cid = _hand_card_id_from_opt(o, cur, me)
            score = -1
            if cid == ABRA_ID:
                score = 5
            elif cid == DUNSPARCE_ID:
                score = 3
            elif cid == KADABRA_ID:
                score = 2
            elif cid == ALAKAZAM_ID:
                score = 1
            if score > best_id:
                best_id, best = score, i
        if best >= 0:
            _rule_fires[3] += 1
            return [best]
        return None  # let IL tiebreak handle

    if ctx == CTX_SETUP_BENCH:
        best, best_id = -1, -1
        for i, o in enumerate(opts):
            if _safe_int(o.get("type")) != OPT_CARD:
                continue
            cid = _hand_card_id_from_opt(o, cur, me)
            score = -1
            if cid == ABRA_ID:
                score = 5
            elif cid == DUNSPARCE_ID:
                score = 3
            if score > best_id:
                best_id, best = score, i
        if best >= 0:
            _rule_fires[3] += 1
            return [best]
        return None

    # ── Rule 5 follow-up: TO_ACTIVE / SWITCH prompt — promote Alakazam ───────
    if ctx in (CTX_TO_ACTIVE, CTX_SWITCH):
        # Look for a CARD option whose target pokemon id == 743 (bench Alakazam).
        for i, o in enumerate(opts):
            if _safe_int(o.get("type")) not in (OPT_CARD,):
                continue
            pidx = _safe_int(o.get("playerIndex"), me)
            area = _safe_int(o.get("area"))
            idx = _safe_int(o.get("index"))
            pl = (cur.get("players") or [{}, {}])[pidx] if pidx < len(cur.get("players") or []) else my
            if area == 5:  # bench
                bn = pl.get("bench") or []
                if 0 <= idx < len(bn) and _pk_id(bn[idx]) == ALAKAZAM_ID:
                    _rule_fires[5] += 1
                    return [i]
            elif area == 4:  # active of a player
                act = pl.get("active") or []
                if act and _pk_id(act[0]) == ALAKAZAM_ID:
                    _rule_fires[5] += 1
                    return [i]
        # No Alakazam to promote — fall through to IL tiebreak.

    # ── Rule 7 follow-up: Boss's Orders gust target — highest-HP non-Cubchoo ─
    if ctx in (CTX_TO_ACTIVE, CTX_SWITCH) and opp_active_id == CUBCHOO_ID:
        # (handled above if Alakazam; here we are choosing an OPPONENT gust target)
        # We want to gust a non-506 opponent pokemon. Options are CARD with opp bench.
        best_i, best_hp = -1, -1
        for i, o in enumerate(opts):
            if _safe_int(o.get("type")) != OPT_CARD:
                continue
            pidx = _safe_int(o.get("playerIndex"), me)
            area = _safe_int(o.get("area"))
            idx = _safe_int(o.get("index"))
            if pidx == me:
                continue  # own pokemon — not a gust target
            pl = (cur.get("players") or [{}, {}])[pidx] if pidx < len(cur.get("players") or []) else opp
            pk = None
            if area == 5:
                bn = pl.get("bench") or []
                if 0 <= idx < len(bn):
                    pk = bn[idx]
            elif area == 4:
                act = pl.get("active") or []
                pk = act[0] if act else None
            pid = _pk_id(pk)
            if pid == 0 or pid == CUBCHOO_ID:
                continue
            hp = _safe_int((pk or {}).get("hp"))
            if hp > best_hp:
                best_hp, best_i = hp, i
        if best_i >= 0:
            _rule_fires[7] += 1
            return [best_i]

    # ── MAIN-context rules (2-8) ─────────────────────────────────────────────
    if ctx != CTX_MAIN:
        # Other contexts (TO_HAND prize/search, DISCARD, etc.) → IL tiebreak.
        return None

    # Pre-scan option types
    attack_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_ATTACK]
    evolve_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_EVOLVE]
    attach_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_ATTACH]
    retreat_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_RETREAT]
    play_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_PLAY]
    ability_opts = [i for i, o in enumerate(opts) if _safe_int(o.get("type")) == OPT_ABILITY]

    # ── Rule 2: Lethal closure (TOP) ─────────────────────────────────────────
    if (alakazam_active and attack_opts and opp_active_hp > 0
            and POWERFUL_HAND_DMG_PER_CARD * hand_count >= opp_active_hp):
        _rule_fires[2] += 1
        return [attack_opts[0]]

    # ── Rule 3: Setup — evolve toward Alakazam ───────────────────────────────
    # 3a: EVOLVE with hand card 743 (Kadabra→Alakazam) or 1079 (Rare Candy → Alakazam)
    evo_to_ala = []
    for i in evolve_opts:
        o = opts[i]
        cid = _hand_card_id(hand_list, _safe_int(o.get("index")))
        if cid in (ALAKAZAM_ID, RARE_CANDY_ID):
            # target pokemon id should be 742 (Kadabra) for 743, or 741 (Abra) for 1079
            tgt = _target_pk(my_active, my_bench,
                             _safe_int(o.get("inPlayArea")),
                             _safe_int(o.get("inPlayIndex")))
            tgt_id = _pk_id(tgt)
            # Rare Candy targets a Basic (Abra 741) → Alakazam; 743 targets Kadabra 742.
            # Accept either (legality guaranteed by engine).
            if (cid == RARE_CANDY_ID and tgt_id == ABRA_ID) or \
               (cid == ALAKAZAM_ID and tgt_id in (KADABRA_ID, ABRA_ID)):
                evo_to_ala.append((i, _safe_int(o.get("inPlayArea")) == 4))
    if evo_to_ala:
        # Prefer one targeting the ACTIVE (inPlayArea==4) so Alakazam lands active.
        evo_to_ala.sort(key=lambda t: (not t[1], t[0]))
        _rule_fires[3] += 1
        return [evo_to_ala[0][0]]

    # 3b: EVOLVE with hand card 742 (Abra→Kadabra, step toward Alakazam)
    for i in evolve_opts:
        o = opts[i]
        cid = _hand_card_id(hand_list, _safe_int(o.get("index")))
        if cid == KADABRA_ID:
            tgt = _target_pk(my_active, my_bench,
                             _safe_int(o.get("inPlayArea")),
                             _safe_int(o.get("inPlayIndex")))
            if _pk_id(tgt) == ABRA_ID:
                _rule_fires[3] += 1
                return [i]

    # ── Rule 4: Energy attach to Alakazam (if it lacks Psychic) ──────────────
    if alakazam_in_play:
        # Find an Alakazam (active or bench) without Psychic energy.
        ala_needs_energy = False
        ala_targets = []  # (option_index, is_active)
        for i in attach_opts:
            o = opts[i]
            energy_cid = _hand_card_id(hand_list, _safe_int(o.get("index")))
            if energy_cid not in PSYCHIC_ENERGY_IDS:  # only basic/telepath psychic pay {P}
                continue
            tgt = _target_pk(my_active, my_bench,
                             _safe_int(o.get("inPlayArea")),
                             _safe_int(o.get("inPlayIndex")))
            if _pk_id(tgt) == ALAKAZAM_ID and not _pk_has_psychic(tgt):
                ala_targets.append((i, _safe_int(o.get("inPlayArea")) == 4))
                ala_needs_energy = True
        if ala_needs_energy and ala_targets:
            ala_targets.sort(key=lambda t: (not t[1], t[0]))  # prefer active Alakazam
            _rule_fires[4] += 1
            return [ala_targets[0][0]]

    # ── Rule 5: Get Alakazam active (retreat, then promote) ──────────────────
    bench_ala = _bench_alakazam_index(my_bench)
    if (not alakazam_active and bench_ala >= 0 and retreat_opts
            and not evo_to_ala):  # no evolve-to-Alakazam available this turn
        # Guard: only promote if the bench Alakazam can attack (has Psychic energy),
        # OR the current active is not our setup-line mon and is about to die.
        bench_ala_pk = my_bench[bench_ala] if bench_ala < len(my_bench) else None
        ala_can_attack = _pk_has_psychic(bench_ala_pk)
        active_low_hp = (my_active is None or
                         _safe_int((my_active or {}).get("hp")) <= 20)
        active_is_setup_line = my_active_id in (ABRA_ID, KADABRA_ID, ALAKAZAM_ID)
        if ala_can_attack or (active_low_hp and not active_is_setup_line):
            _rule_fires[5] += 1
            return [retreat_opts[0]]  # follow-up TO_ACTIVE prompt handled above

    # ── Rule 6: Hand-ramp before attacking (Alakazam active + energy, not lethal) ─
    if alakazam_active and _pk_has_psychic(my_active) and opp_active_hp > 0:
        not_lethal = POWERFUL_HAND_DMG_PER_CARD * hand_count < opp_active_hp
        if not_lethal:
            # Rank hand-grow options. Order: supporters (Hilda/Dawn) > Poke Pad >
            # Alakazam/Kadabra Psychic Draw ability > Dudunsparce Run Away Draw >
            # ATTACH Enriching Energy 13.
            ramp_candidates: list[tuple[int, int]] = []  # (rank, option_index)
            for i in play_opts:
                o = opts[i]
                cid = _hand_card_id(hand_list, _safe_int(o.get("index")))
                if cid in HAND_GROW_SUPPORTERS:
                    ramp_candidates.append((0, i))
                elif cid in HAND_GROW_ITEMS:  # Poke Pad
                    ramp_candidates.append((1, i))
            for i in ability_opts:
                o = opts[i]
                pk = _ability_pk(my_active, my_bench, o)
                pid = _pk_id(pk)
                if pid in (ALAKAZAM_ID, KADABRA_ID):  # Psychic Draw
                    ramp_candidates.append((2, i))
                elif pid == DUDUNSPARCE_ID:  # Run Away Draw
                    ramp_candidates.append((3, i))
            for i in attach_opts:
                o = opts[i]
                ecid = _hand_card_id(hand_list, _safe_int(o.get("index")))
                if ecid == ENRICHING_ENERGY_ID:
                    ramp_candidates.append((4, i))
            if ramp_candidates:
                ramp_candidates.sort(key=lambda t: (t[0], t[1]))
                _rule_fires[6] += 1
                return [ramp_candidates[0][1]]

    # ── Rule 7: Cubchoo-lock pivot (Boss's Orders to gust a non-Cubchoo) ─────
    if opp_active_id == CUBCHOO_ID:
        for i in play_opts:
            o = opts[i]
            cid = _hand_card_id(hand_list, _safe_int(o.get("index")))
            if cid == BOSS_ID:
                _rule_fires[7] += 1
                return [i]  # follow-up target prompt handled above / by IL

    # ── Rule 8: Attack fallback ──────────────────────────────────────────────
    if attack_opts:
        # If Alakazam active + energy → take Powerful Hand (chip / take a prize).
        # Otherwise attack with whatever the active can do (don't stall into deck-out).
        _rule_fires[8] += 1
        return [attack_opts[0]]

    return None  # no rule matched → IL tiebreak


def _hand_card_id_from_opt(o, cur, me) -> int:
    """For a CARD option pointing at a hand card (area==2)."""
    area = _safe_int(o.get("area"))
    if area != 2:
        return 0
    pidx = _safe_int(o.get("playerIndex"), me)
    idx = _safe_int(o.get("index"))
    players = cur.get("players") or [{}, {}]
    pl = players[pidx] if pidx < len(players) else {}
    h = pl.get("hand") or []
    if 0 <= idx < len(h):
        return _safe_int((h[idx] or {}).get("id")) if isinstance(h[idx], dict) else 0
    return 0


# ── IL tiebreak (rule 9) ─────────────────────────────────────────────────────
def _il_decide(obs_dict: dict, n_opts: int, min_count: int, max_count: int) -> list[int] | None:
    if _ranker is None or not _features_ok:
        return None
    try:
        X = build_feature_matrix(obs_dict)
        if X.shape[0] != n_opts:
            return None
        scores = _ranker.predict(X)
        order = [int(i) for i in np.argsort(scores)[::-1]]
        order = [i for i in order if 0 <= i < n_opts]
        k = max(min_count, 1)
        k = min(k, max_count, n_opts)
        if not order:
            return None
        return order[:k]
    except Exception:
        return None


# ── Public agent (never-throw) ───────────────────────────────────────────────
def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point.

    Priority:
      0. Deck select (obs.select is None) → 60-id deck.
      1. Never-throw wrapper.
      2-8. Hand-written heuristic rules (setup→closure plan).
      9. IL ranker tiebreak (only for uncovered decisions).
      10. Trivial fallback (first minCount indices).
    """
    global _n_model_decisions, _n_ko_guard, _n_heuristic_fallback
    global _n_trivial_fallback, _exception_count

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        _n_model_decisions += 1
        return my_deck

    opts = sel.get("option") or []
    n_options = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 1. Heuristic (rules 2-8)
    try:
        choice = _heuristic(obs_dict, opts, sel)
        if choice is not None and choice:
            if (all(0 <= i < n_options for i in choice)
                    and len(set(choice)) == len(choice)
                    and len(choice) >= min_count
                    and len(choice) <= max_count):
                _n_model_decisions += 1
                if _rule_fires.get(2, 0) and choice:
                    pass  # rule 2 already counted in _rule_fires
                return choice
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(f"[v017] heuristic exception #{_exception_count}: {exc}", file=sys.stderr)

    # 2. IL tiebreak (rule 9)
    try:
        il_choice = _il_decide(obs_dict, n_options, min_count, max_count)
        if il_choice and all(0 <= i < n_options for i in il_choice) and len(il_choice) >= min_count:
            _n_heuristic_fallback += 1
            _rule_fires[9] += 1
            return il_choice
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 6:
            print(f"[v017] IL exception #{_exception_count}: {exc}", file=sys.stderr)

    # 3. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))