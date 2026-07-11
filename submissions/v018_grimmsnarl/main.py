"""
v018_grimmsnarl — IL agent: Marnie's Grimmsnarl ex / Munkidori toolbox + LightGBM ranker.
Single-expert BC clone of kazuki0123 #3 (teamId 16441533, subId 54203196, mu=1195, wr=0.67).
MIDRANGE deck (same profile as the proven Crustle clones v007/v009a, 906-930 live):
72/72 sampled expert attacks were Grimmsnarl ex "Shadow Bullet".
Config C6: lr=0.005, num_leaves=15, min_child_samples=50, 300 trees, early-stop=OFF.
kazuki0123 Grimmsnarl episodes: 73, MAIN winner decisions: 2335.
Offline val MAIN top-1: 0.7613 (checkpoint only — NOT a pass/fail gate).

Architecture:
  - Load IL ranker (il_grimmsnarl.txt) at import time.
  - Deck: Grimmsnarl ex toolbox (60 cards, legal; see deck.csv — kazuki0123's exact list).
  - On deck-selection step (obs.select is None): return the 60-card deck.
  - On all other steps:
      1. ACTIVE-PROMOTION GUARD (runs BEFORE the IL, added 2026-07-04):
         Whenever the select context is TO_ACTIVE (4) or SWITCH (3) — i.e. we
         must choose a bench Pokemon to become the new Active (forced promotion
         after a KO, or a voluntary switch) — deterministically pick the best
         bench Pokemon instead of asking the IL ranker. DIAGNOSIS: the ranker
         was trained ONLY on MAIN-context (ctx==0) decisions from the human
         replay data, so this context is completely out-of-distribution. Traced
         evidence (6 games vs v001_random, 37 forced-promotion decisions):
         the ranker NEVER once promoted the healthy benched Grimmsnarl ex
         (id 648, 320 HP) when it was available — it consistently promoted
         fragile support Pokemon instead (Dunsparce 305, Impidimp/Morgrem
         pre-evolutions, Munkidori 112), leaving the active slot occupied by
         a mon that can't threaten lethal, or repeatedly re-KO'd. This
         directly explains the 15/15 vs-random loss pattern (empty active /
         weak active at end-state while a healthy Grimmsnarl sat on bench).
         Guard priority: Grimmsnarl ex (648, HP>0) > Morgrem (647, real
         attacker) > Morpeko (649, real attacker) > any other unlisted mon >
         known fragile/support mons (Dunsparce 305, Dudunsparce 66,
         Impidimp 646, Munkidori 112 — no real attack). Tie-break: highest HP.
      2. HARD LETHAL-KO GUARD (anti-whiff, runs BEFORE the IL):
         If an ATTACK option's listed damage >= opp active remaining HP, take it
         immediately. Grimmsnarl ex's "Shadow Bullet" (cost {D}{D}) does 180
         damage to the opponent's Active (+30 to a Benched Pokemon, no weakness/
         resistance on bench damage) — its only attack. This guard mirrors the
         v016_alakazam lesson: BC of a Pokemon-line deck can whiff a clean kill
         the ranker was uncertain about; a cheap deterministic KO check never hurts.
      3. ENERGY-ATTACH GUARD (runs AFTER the lethal-KO guard, BEFORE the IL,
         added 2026-07-04): Traced evidence (scratchpad/trace_v018_vs_crustle.py,
         10 games vs v007_crustle) found Grimmsnarl ex 648 reaches Active with
         full HP by turn 6-9 but sits at ZERO energy the ENTIRE game -> never
         attacks (attacked in only 2/10 games, turns 36/50). Shadow Bullet
         costs {D}{D}; the IL ranker mis-prioritizes the ATTACH-{D}-to-active
         option against competing MAIN options (draw/supporter/bench-evolve),
         so the active attacker never gets its attack-cost energy. Same
         "ranker mis-ranks an under-represented option class" failure mode as
         the promotion guard above. Fix: if my active is Grimmsnarl ex 648
         and it has < 2 {D} energy attached, and an ATTACH option exists that
         puts a Basic {D} Energy (id 7) onto the ACTIVE Pokemon, take it
         immediately (bypass the ranker). Don't over-attach: if active
         already has >= 2 {D} (enough for Shadow Bullet), skip this guard and
         let the ranker / other guards proceed (e.g. so it can attack or set
         up bench).
      3.5 ANTI-CRUSTLE MUNKIDORI ADRENA-BRAIN SNIPE GUARD (added 2026-07-04):
         DIAGNOSIS: vs v007/v009a Crustle, Shadow Bullet's 180 dmg does NOT
         OHKO Crustle's ~270 HP attackers -> Grimmsnarl trades instead of
         cleanly KOing. Munkidori's (112) [Ability] Adrena-Brain ("move up to
         3 damage counters from 1 of your Pokemon to 1 of your opponent's
         Pokemon", requires a {D} energy on Munkidori) can convert an
         otherwise-nonlethal turn into a KO: it moves 30 dmg FROM a damaged
         ally (e.g. the active Grimmsnarl, which accumulates damage from
         Crustle's counterattacks) ONTO the opponent's active. Verified via
         data/EN_Card_Data.csv (card 112: "...move up to 3 damage counters
         from 1 of your Pokemon to 1 of your opponent's Pokemon" -> confirmed
         it moves TO the opponent, so the snipe guard is valid) and via a
         replay scan of kazuki0123's own Adrena-Brain usages (5 real examples
         in data/episodes/all_teams/episode-83558495-replay.json), which
         confirmed the exact 3-step engine sequence for this ability:
           step 1 (this guard): ctx=MAIN, choose the OPT_ABILITY(=10) option
             whose (area,index) resolves to Munkidori (112).
           step 2: ctx=REMOVE_DAMAGE_COUNTER(=16), OPT_CARD options for which
             of MY Pokemon to take damage counters FROM.
           step 3: ctx=REMOVE_DAMAGE_COUNTER_COUNT(=40), OPT_NUMBER(=0)
             options for HOW MANY counters (0..3) to move.
           step 4: ctx=DAMAGE_COUNTER(=13), OPT_CARD options for which
             opponent Pokemon to place the counters ON (the snipe target).
         Fires ONLY when it converts to a guaranteed KO this turn (never
         "wastes" the once-per-turn ability): (a) opp active HP <= available
         snipe damage (direct kill), or (b) opp active HP is in the band
         (180, 180+available_snipe] AND my active Grimmsnarl already has its
         {D}{D} Shadow Bullet cost ready, so the ability's damage plus this
         turn's Shadow Bullet combine to KO (the existing hard-lethal-KO
         guard then fires the attack on the very next MAIN decision once the
         opponent's HP has dropped). A tiny state machine (`_adrena_state`)
         carries the guard through steps 2-4 since they are forced
         continuations of this guard's own step-1 choice, not independent
         MAIN decisions the IL ranker should see (same "OOD context" failure
         class as the promotion guard).
         SKIPPED: a companion "retreat to preserve energy investment" guard
         was evaluated (CSV: Grimmsnarl ex 648 retreat cost = 2) and NOT
         added. Reason: the existing energy-attach guard (item 3) caps
         Grimmsnarl's {D} energy at exactly 2 (never over-attaches), and
         retreating pays its cost by discarding attached energy 1-for-1 -- so
         retreating would always discard 100% of the "investment" being
         "preserved", making the guard's own premise vacuous. The only real
         benefit left (denying the opponent a prize) requires guessing
         whether Grimmsnarl is "about to be KO'd", which is unobservable
         (opponent hand/attack choice is hidden) -- any HP-threshold trigger
         would be a speculative heuristic that risks forfeiting a turn's
         attack (the promoted bench mon usually has no energy to swing) for
         uncertain payoff. Given the instruction to keep guards narrow and
         not risk the random/Alakazam wins, this guard was skipped.
      4. IL ranker: build per-option feature matrix (features_grimmsnarl.py) ->
         score with model -> pick best.
      5. Trivial fallback (never throws): first minCount indices.

Kaggle loads main.py via exec() WITHOUT __file__ -> guarded _DIR resolution.
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

import numpy as np

# ── Instrumentation counters ─────────────────────────────────────────────────
_n_model_decisions: int = 0
_n_ko_guard: int = 0
_n_promo_guard: int = 0
_n_energy_guard: int = 0
_n_adrena_trigger: int = 0   # number of times the Adrena-Brain snipe was initiated (step 1)
_n_adrena_steps: int = 0     # number of forced-continuation steps handled (steps 2-4)
_n_trivial_fallback: int = 0
_exception_count: int = 0

# ── Constants ────────────────────────────────────────────────────────────────
OPT_NUMBER = 0
OPT_CARD = 3
OPT_ATTACH = 8
OPT_ABILITY = 10
OPT_ATTACK = 13

# SelectContext values relevant to the guards (mirrors cg/api.py)
CTX_MAIN = 0        # Main selection (PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END, ...).
CTX_SWITCH = 3      # Select the Pokemon to swap with the one in your Active Spot.
CTX_TO_ACTIVE = 4   # Select the Pokemon to put into your Active Spot (forced promotion).
CTX_DAMAGE_COUNTER = 13              # Select the Pokemon to PLACE damage counters ON (snipe target).
CTX_REMOVE_DAMAGE_COUNTER = 16       # Select the Pokemon to remove damage counters FROM (snipe source).
CTX_REMOVE_DAMAGE_COUNTER_COUNT = 40 # Select HOW MANY damage counters to remove/move (0..3).

# AreaType values relevant to the energy-attach / Adrena-Brain guards (mirrors cg/api.py)
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5

# Card IDs (Marnie's Grimmsnarl ex toolbox)
GRIMMSNARL_ID = 648      # Stage 2, 320 HP, "Shadow Bullet" 180 dmg — the win-con attacker.
MORGREM_ID = 647         # Stage 1, 100 HP, "Corkscrew Punch" 60 dmg — real (if weaker) attacker.
MORPEKO_ID = 649         # Basic, 70 HP, "Spiky Wheel" scales with {D} energy — real attacker.
MUNKIDORI_ID = 112       # Basic, 110 HP, [Ability] Adrena-Brain — damage-counter snipe (needs {D}).
BASIC_D_ENERGY_ID = 7    # Basic {D} (Darkness) Energy — Shadow Bullet costs {D}{D}.

# Adrena-Brain snipe sequence state (module-level; the sub-decisions at steps
# 2-4 are forced continuations of this guard's own step-1 choice, not
# independent MAIN decisions — see module docstring item 3.5).
_adrena_state: str | None = None  # None | "await_from" | "await_count" | "await_target"
# Known fragile/support mons with no meaningful attack (deprioritize as promotion target
# whenever a real attacker is available on bench).
FRAGILE_SUPPORT_IDS = {
    305,   # Dunsparce — 0-dmg switch utility
    66,    # Dudunsparce — draw-then-shuffle-self utility
    646,   # Marnie's Impidimp — 0-dmg draw utility (Grimmsnarl pre-evo, setup only)
    112,   # Munkidori — ability-only (damage-counter mover), no attack move
}


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

# ── Load LightGBM model ──────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _model_path = _find_file("il_grimmsnarl.txt")
    _ranker: "lgb.Booster | None" = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[v018] WARNING: model load failed: {_load_err}", file=sys.stderr)

# ── Feature extraction ───────────────────────────────────────────────────────
try:
    from features_grimmsnarl import build_feature_matrix  # noqa: E402
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[v018] WARNING: features_grimmsnarl.py import failed: {_feat_err}", file=sys.stderr)

# ── Card damage table (for the lethal-KO guard; independent of features module) ──
try:
    import re
    import pandas as pd

    _card_csv_path = _find_file("EN_Card_Data.csv")
    _df = pd.read_csv(_card_csv_path)
    _df.columns = _df.columns.str.strip()

    def _parse_dmg(s):
        if s is None or (isinstance(s, float) and s != s):  # NaN check
            return 0
        try:
            return int(re.sub(r"[^0-9]", "", str(s)) or "0")
        except Exception:
            return 0

    _DAMAGE_TABLE: dict[int, list[int]] = {}
    for _cid, _grp in _df.groupby("Card ID"):
        # Exclude ability rows (e.g. "[Ability] Punk Up") — in-game ATTACK
        # options only correspond to non-ability move rows, and attack_rank
        # must index into a list built the same way (see features_grimmsnarl.py).
        _rows = [
            r for _, r in _grp.iterrows()
            if not str(r.get("Move Name") or "").strip().startswith("[Ability]")
        ]
        _DAMAGE_TABLE[int(_cid)] = [_parse_dmg(r.get("Damage")) for r in _rows]
    _damage_table_ok = True
except Exception as _dmg_err:
    _DAMAGE_TABLE = {}
    _damage_table_ok = False
    print(f"[v018] WARNING: damage table load failed: {_dmg_err}", file=sys.stderr)


# ── Active-promotion guard ───────────────────────────────────────────────────
def _promo_priority(pk_id: int, hp: int) -> tuple[int, int]:
    """(tier, hp) sort key — higher tier wins, hp is the tie-break within a tier."""
    if pk_id == GRIMMSNARL_ID and hp > 0:
        return (3, hp)
    if pk_id in (MORGREM_ID, MORPEKO_ID) and hp > 0:
        return (2, hp)
    if pk_id in FRAGILE_SUPPORT_IDS:
        return (0, hp)
    # Unlisted mon (e.g. an evolving Impidimp mid-toolbox, or a tech card) —
    # treat as medium priority: better than known-fragile, worse than a real attacker.
    return (1, hp)


def _promotion_guard(obs_dict: dict) -> list[int] | None:
    """Forced-promotion / switch guard (ctx TO_ACTIVE / SWITCH).

    Bypasses the IL ranker entirely for this context (OOD — ranker trained
    only on MAIN-context decisions; see module docstring for trace evidence).
    Deterministically picks the best available bench Pokemon:
    Grimmsnarl ex (healthy) > other real attacker > unlisted mon > fragile/support.
    Returns None if the option list doesn't look like the expected bench-CARD
    shape (falls through to IL / trivial fallback — never-throw preserved).
    """
    try:
        sel = obs_dict.get("select") or {}
        ctx = int(sel.get("context") or -1)
        if ctx not in (CTX_SWITCH, CTX_TO_ACTIVE):
            return None
        opts = sel.get("option") or []
        if not opts:
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_bench = players[me].get("bench") or []

        best_i = -1
        best_key = (-1, -1)
        for i, o in enumerate(opts):
            if int(o.get("type") or -1) != OPT_CARD:
                continue
            if int(o.get("area") if o.get("area") is not None else -1) != 5:  # AreaType.BENCH
                continue
            opt_player = o.get("playerIndex")
            if opt_player is not None and int(opt_player) != me:
                continue
            idx = int(o.get("index") or -1)
            if not (0 <= idx < len(my_bench)):
                continue
            pk = my_bench[idx]
            if not isinstance(pk, dict):
                continue
            pk_id = int(pk.get("id") or 0)
            hp = int(pk.get("hp") or 0)
            key = _promo_priority(pk_id, hp)
            if key > best_key:
                best_key, best_i = key, i
        if best_i >= 0:
            return [best_i]
        return None
    except Exception:
        return None


# ── Lethal-KO guard ──────────────────────────────────────────────────────────
def _ko_guard(obs_dict: dict) -> list[int] | None:
    """If any ATTACK option's listed damage >= opp active remaining HP,
    return that option index immediately. Else None.

    Generic guard (not hard-coded to Shadow Bullet's 180 dmg): reads the
    attacking pokemon's damage list from EN_Card_Data.csv by attack rank,
    same convention as features_grimmsnarl.build_feature_matrix.
    """
    if not _damage_table_ok:
        return None
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        if not opts:
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]
        opp_pl = players[1 - me]
        my_active = (my_pl.get("active") or [None])[0]
        opp_active = (opp_pl.get("active") or [None])[0]
        if my_active is None or opp_active is None:
            return None
        my_active_id = int(my_active.get("id") or 0)
        opp_hp = int(opp_active.get("hp") or 0)
        if opp_hp <= 0:
            return None
        damages = _DAMAGE_TABLE.get(my_active_id) or []
        if not damages:
            return None
        attack_rank = 0
        for i, opt in enumerate(opts):
            if int(opt.get("type") or 0) != OPT_ATTACK:
                continue
            dmg = damages[attack_rank] if attack_rank < len(damages) else (damages[-1] if damages else 0)
            attack_rank += 1
            if dmg > 0 and dmg >= opp_hp:
                return [i]
        return None
    except Exception:
        return None


# ── Energy-attach guard ──────────────────────────────────────────────────────
def _energy_guard(obs_dict: dict) -> list[int] | None:
    """MAIN-context guard: force-attach {D} energy onto an active Grimmsnarl ex
    that still lacks its {D}{D} Shadow Bullet attack cost.

    DIAGNOSIS (2026-07-04, scratchpad/trace_v018_vs_crustle.py, 10 games vs
    v007_crustle): Grimmsnarl ex 648 reaches Active with full HP by turn 6-9
    but sits at ZERO energy the ENTIRE game -> never attacks (attacked in
    only 2/10 games, turns 36/50). The IL ranker (trained only on human
    MAIN-context decisions) mis-prioritizes the ATTACH-{D}-to-active option
    against competing MAIN options (draw/supporter/bench-evolve), so the
    active attacker never gets energized. Same failure class as the
    promotion guard above (ranker mis-ranks a option-type it under-weights).

    Condition: ctx == MAIN, my active is Grimmsnarl ex (648) with < 2 {D}
    energy already attached, AND an ATTACH option exists in the option list
    that puts a Basic {D} Energy (card id 7) from hand onto the ACTIVE
    Pokemon. Returns that option's index immediately (bypasses the ranker).

    Don't over-attach: if the active already has >= 2 {D} energy (enough for
    Shadow Bullet), return None so the ranker / other guards proceed
    normally (e.g. so it can attack this turn, or set up the bench instead).

    Returns None if the condition doesn't hold or no qualifying option is
    found (falls through to the IL ranker / trivial fallback — never-throw
    preserved).
    """
    try:
        sel = obs_dict.get("select") or {}
        ctx = int(sel.get("context") if sel.get("context") is not None else -1)
        if ctx != CTX_MAIN:
            return None
        opts = sel.get("option") or []
        if not opts:
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]
        my_active = (my_pl.get("active") or [None])[0]
        if not my_active:
            return None
        if int(my_active.get("id") or 0) != GRIMMSNARL_ID:
            return None

        dark_energy_on_active = sum(
            1 for e in (my_active.get("energies") or []) if int(e) == BASIC_D_ENERGY_ID
        )
        if dark_energy_on_active >= 2:
            return None  # already has Shadow Bullet's {D}{D} cost -- don't over-attach

        my_hand = my_pl.get("hand") or []
        for i, o in enumerate(opts):
            if int(o.get("type") or -1) != OPT_ATTACH:
                continue
            area = int(o.get("area") if o.get("area") is not None else -1)
            idx = int(o.get("index") if o.get("index") is not None else -1)
            if area != AREA_HAND or not (0 <= idx < len(my_hand)):
                continue
            card = my_hand[idx]
            if not isinstance(card, dict) or int(card.get("id") or 0) != BASIC_D_ENERGY_ID:
                continue
            in_area = int(o.get("inPlayArea") if o.get("inPlayArea") is not None else -1)
            if in_area != AREA_ACTIVE:
                continue
            return [i]
        return None
    except Exception:
        return None


# ── Adrena-Brain snipe guard (Munkidori 112) ─────────────────────────────────
def _resolve_pokemon(player_dict: dict, area: int, idx: int) -> dict | None:
    """Resolve a Pokemon dict from (area, index) — same convention used by
    features_grimmsnarl._get_pokemon_from_obs. area: 4=ACTIVE, 5=BENCH."""
    if area == AREA_ACTIVE:
        active = player_dict.get("active") or []
        return active[0] if active else None
    if area == AREA_BENCH:
        bench = player_dict.get("bench") or []
        return bench[idx] if 0 <= idx < len(bench) else None
    return None


def _damage_on(pk: dict | None) -> int:
    """Damage counters worth of HP missing (maxHp - hp), floored at 0."""
    if not isinstance(pk, dict):
        return 0
    hp = int(pk.get("hp") or 0)
    maxhp = int(pk.get("maxHp") or hp)
    return max(0, maxhp - hp)


def _adrena_trigger_guard(obs_dict: dict) -> list[int] | None:
    """MAIN-context: decide whether to INITIATE the Adrena-Brain snipe combo.

    Only fires when it converts to a guaranteed KO this turn:
      (a) opp active HP <= available snipe damage (direct kill), or
      (b) opp active HP in (180, 180 + available_snipe] AND my active
          Grimmsnarl ex already has its {D}{D} Shadow Bullet cost ready
          (so the hard-lethal-KO guard finishes the KO with the attack on
          the very next MAIN decision, once this snipe lands).
    Available snipe damage = 10 * min(3, max damage-counters on any one of
    my Pokemon) — Adrena-Brain moves "up to 3" counters (30 dmg) from ONE
    of my Pokemon. Sets _adrena_state="await_from" and returns the
    OPT_ABILITY option index on success. See module docstring item 3.5.
    """
    global _adrena_state
    try:
        sel = obs_dict.get("select") or {}
        ctx = int(sel.get("context") if sel.get("context") is not None else -1)
        if ctx != CTX_MAIN:
            return None
        opts = sel.get("option") or []
        if not opts:
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]
        opp_pl = players[1 - me]
        opp_active = (opp_pl.get("active") or [None])[0]
        if not opp_active:
            return None
        opp_hp = int(opp_active.get("hp") or 0)
        if opp_hp <= 0:
            return None

        ability_idx = None
        for i, o in enumerate(opts):
            if int(o.get("type") or -1) != OPT_ABILITY:
                continue
            area = int(o.get("area") if o.get("area") is not None else -1)
            idx = int(o.get("index") if o.get("index") is not None else -1)
            src = _resolve_pokemon(my_pl, area, idx)
            if src and int(src.get("id") or 0) == MUNKIDORI_ID:
                ability_idx = i
                break
        if ability_idx is None:
            return None

        my_active = (my_pl.get("active") or [None])[0]
        candidates = ([my_active] if my_active else []) + list(my_pl.get("bench") or [])
        max_counters = 0
        for pk in candidates:
            max_counters = max(max_counters, min(3, _damage_on(pk) // 10))
        if max_counters <= 0:
            return None  # nothing to snipe with -- don't waste the once-per-turn ability
        available_snipe = max_counters * 10

        if opp_hp <= available_snipe:
            _adrena_state = "await_from"
            return [ability_idx]

        if my_active and int(my_active.get("id") or 0) == GRIMMSNARL_ID:
            dark_energy = sum(
                1 for e in (my_active.get("energies") or []) if int(e) == BASIC_D_ENERGY_ID
            )
            if dark_energy >= 2:
                shadow_bullet_dmg = (_DAMAGE_TABLE.get(GRIMMSNARL_ID) or [180])[0] if _damage_table_ok else 180
                if opp_hp > shadow_bullet_dmg and opp_hp <= shadow_bullet_dmg + available_snipe:
                    _adrena_state = "await_from"
                    return [ability_idx]
        return None
    except Exception:
        return None


def _adrena_sequence_guard(obs_dict: dict) -> list[int] | None:
    """Forced-continuation guard for steps 2-4 of an already-initiated
    Adrena-Brain snipe (see _adrena_trigger_guard). Must run BEFORE every
    other guard/ranker while `_adrena_state` is set, since these contexts
    (REMOVE_DAMAGE_COUNTER / _COUNT / DAMAGE_COUNTER) are OOD for the IL
    ranker and not what the promotion/KO/energy guards look for anyway.
    Resets `_adrena_state` to None on completion, desync, or any failure
    (never-throw preserved; a reset just lets the normal chain run for
    this decision instead of guessing).
    """
    global _adrena_state, _n_adrena_steps
    if _adrena_state is None:
        return None
    try:
        sel = obs_dict.get("select") or {}
        ctx = int(sel.get("context") if sel.get("context") is not None else -1)
        opts = sel.get("option") or []
        if not opts:
            _adrena_state = None
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]

        if _adrena_state == "await_from" and ctx == CTX_REMOVE_DAMAGE_COUNTER:
            best_i, best_dmg = None, -1
            for i, o in enumerate(opts):
                if int(o.get("type") or -1) != OPT_CARD:
                    continue
                pidx = o.get("playerIndex")
                if pidx is not None and int(pidx) != me:
                    continue
                area = int(o.get("area") if o.get("area") is not None else -1)
                idx = int(o.get("index") if o.get("index") is not None else -1)
                dmg = _damage_on(_resolve_pokemon(my_pl, area, idx))
                if dmg > best_dmg:
                    best_dmg, best_i = dmg, i
            _adrena_state = "await_count" if best_i is not None else None
            if best_i is not None:
                _n_adrena_steps += 1
                return [best_i]
            return None

        if _adrena_state == "await_count" and ctx == CTX_REMOVE_DAMAGE_COUNTER_COUNT:
            best_i, best_n = None, -1
            for i, o in enumerate(opts):
                # NOTE: OPT_NUMBER == 0, so the usual "x or -1" idiom is unsafe
                # here (0 is falsy) -- use an explicit None check instead.
                if int(o.get("type") if o.get("type") is not None else -1) != OPT_NUMBER:
                    continue
                n = int(o.get("number") if o.get("number") is not None else -1)
                if n > best_n:
                    best_n, best_i = n, i
            _adrena_state = "await_target" if best_i is not None else None
            if best_i is not None:
                _n_adrena_steps += 1
                return [best_i]
            return None

        if _adrena_state == "await_target" and ctx == CTX_DAMAGE_COUNTER:
            opp_idx = 1 - me
            best_i = None
            for i, o in enumerate(opts):
                if int(o.get("type") or -1) != OPT_CARD:
                    continue
                if int(o.get("playerIndex") if o.get("playerIndex") is not None else -1) != opp_idx:
                    continue
                area = int(o.get("area") if o.get("area") is not None else -1)
                if area == AREA_ACTIVE:
                    best_i = i
                    break
            if best_i is None:
                for i, o in enumerate(opts):
                    if int(o.get("type") or -1) == OPT_CARD and int(
                        o.get("playerIndex") if o.get("playerIndex") is not None else -1
                    ) == opp_idx:
                        best_i = i
                        break
            _adrena_state = None
            if best_i is not None:
                _n_adrena_steps += 1
                return [best_i]
            return None

        # Any other context while a snipe is pending is an unexpected desync
        # (e.g. an unforeseen intervening decision) -- reset and let the
        # normal guard chain / ranker handle this decision safely.
        _adrena_state = None
        return None
    except Exception:
        _adrena_state = None
        return None


# ── IL decision ──────────────────────────────────────────────────────────────
def _il_decide(obs_dict: dict) -> list[int]:
    if _ranker is None or not _features_ok:
        raise RuntimeError("model or features unavailable")
    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    if n_opts == 0:
        return []
    X = build_feature_matrix(obs_dict)
    if X.shape[0] != n_opts:
        raise ValueError(f"feature matrix shape mismatch: {X.shape[0]} != {n_opts}")
    scores = _ranker.predict(X)
    order = [int(i) for i in np.argsort(scores)[::-1]]
    order = [i for i in order if 0 <= i < n_opts]
    k = max(min_count, 1)
    k = min(k, max_count, n_opts)
    return order[:k]


# ── Public agent ─────────────────────────────────────────────────────────────
def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point.

    Priority:
      0. Adrena-Brain sequence guard (forced continuation of an already-
         initiated Munkidori snipe -> steps 2-4, see item 3.5 in the module
         docstring). Runs first: these contexts are irrelevant to every
         other guard/ranker anyway, so checking first is a pure no-op when
         no snipe is pending (_adrena_state is None).
      1. Active-promotion guard (forced promotion / switch -> best bench mon).
      2. Lethal-KO guard (any attack that KOs the opponent's active -> take it).
      3. Adrena-Brain trigger guard (initiate the Munkidori snipe when it
         converts to a guaranteed KO this turn, direct or combined with this
         turn's Shadow Bullet).
      4. Energy-attach guard (active Grimmsnarl ex needs {D}{D} -> attach it).
      5. IL ranker.
      6. Trivial fallback (first minCount indices).
    """
    global _n_model_decisions, _n_ko_guard, _n_promo_guard, _n_energy_guard
    global _n_adrena_trigger, _n_adrena_steps, _n_trivial_fallback, _exception_count

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 0. Adrena-Brain sequence guard (forced continuation, no-op if not pending)
    try:
        seq = _adrena_sequence_guard(obs_dict)
        if seq is not None and all(0 <= i < n_options for i in seq):
            return seq
    except Exception:
        pass

    # 1. Active-promotion guard
    try:
        promo = _promotion_guard(obs_dict)
        if promo is not None and all(0 <= i < n_options for i in promo):
            _n_promo_guard += 1
            return promo
    except Exception:
        pass

    # 2. Lethal-KO guard
    try:
        guard = _ko_guard(obs_dict)
        if guard is not None and all(0 <= i < n_options for i in guard):
            _n_ko_guard += 1
            return guard
    except Exception:
        pass

    # 3. Adrena-Brain trigger guard (initiate the Munkidori snipe combo)
    try:
        adrena = _adrena_trigger_guard(obs_dict)
        if adrena is not None and all(0 <= i < n_options for i in adrena):
            _n_adrena_trigger += 1
            return adrena
    except Exception:
        pass

    # 4. Energy-attach guard
    try:
        energy = _energy_guard(obs_dict)
        if energy is not None and all(0 <= i < n_options for i in energy):
            _n_energy_guard += 1
            return energy
    except Exception:
        pass

    # 5. IL model
    try:
        result = _il_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result) and len(result) >= min_count:
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[v018] IL exception #{_exception_count}: {exc}  "
                f"(model={_n_model_decisions} ko={_n_ko_guard} "
                f"fallback={_n_trivial_fallback})",
                file=sys.stderr,
            )

    # 6. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
