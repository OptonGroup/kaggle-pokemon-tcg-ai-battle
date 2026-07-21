"""
v061_grimmsnarl_heuristic — DETERMINISTIC matchup-heuristic pilot for the jiatu.l (mu=1130)
Marnie's Grimmsnarl ex / Munkidori toolbox deck. Track B fallback: forked from
v060_grimmsnarl_jiatu (a LightGBM BC-ranker over this same deck), the ranker (step 5) is
REPLACED by a v034/v036-style greedy deterministic scorer (scoring_grimmsnarl.choose_options,
argmax of score_option over the option list). Motivation: our best LIVE agents (v034 mu=865.7 /
v036 mu=827.3) are deterministic matchup-heuristics, NOT RL/BC; v060's ranker plateaued
(bimodal — Alakazam 0.70-0.81, tanks 0.04-0.19; offline top-1 only 0.541). This ports that
proven method onto the jiatu.l deck. See DESIGN.md for the full scoring spec and the Phase-1
guard edits.

Architecture split (two representations of the SAME engine dict):
  - GUARDS (below) run on the raw obs_dict with integer constants — ported verbatim from
    v060 (with 3 edits, see DESIGN.md Phase-1): they cover the OOD/forced contexts a legible
    scorer can't rank (Adrena-Brain sequence, forced promotion/switch, lethal-KO,
    energy-attach, Adrena-Brain trigger). They run FIRST.
  - SCORER (scoring_grimmsnarl) runs on the typed obs = to_observation_class(obs_dict) with
    cg.api enums — reached only when no guard fires. Option index i is identical in both.

Deck (jiatu.l, 60): Grimmsnarl ex 648 / Morgrem 647 / Impidimp 646 (only real attackers) +
Munkidori 112 (Adrena-Brain engine) + Snorunt 860 / Froslass 104 (Freezing-Shroud engine) +
Poffin/Poké Pad/Lillie/Petrel/Rare Candy/Night Stretcher/Unfair Stamp/Boss/Dawn/Xerosic/
Pokégear + Spikemuth Gym + 10× Basic {D}. The LightGBM model, features_grimmsnarl_v2, pandas
and the bundled EN_Card_Data.csv are all DROPPED — three fewer live-failure surfaces.

Architecture:
  - Load IL ranker (il_grimmsnarl_v2.txt) at import time.
  - Deck: Grimmsnarl ex toolbox (60 cards, legal; see deck.csv — jiatu.l's exact list).
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
         Guard priority: Grimmsnarl ex (648, HP>0) > Morgrem (647)/Morpeko
         (649)/Froslass (104, v2 addition — "Frost Smash" 60 dmg) real
         attackers > any other unlisted mon > known fragile/support mons
         (Dunsparce 305, Dudunsparce 66, Impidimp 646, Munkidori 112 — no
         real attack). Tie-break: highest HP.
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
      3.6 ANTI-TANK GUARD PAIR vs Crustle/Archaludon (added 2026-07-21):
         DIAGNOSIS (results/gates/v060_diag_gate.json, 2026-07-21): v060 gates
         0.70-0.81 WR vs Alakazam (v016/v017) but only 0.037-0.063 vs the
         Crustle wall (v009a) and Archaludon ex beatdown (v044/v049) -- a
         mechanical hole, not noise. Mined it directly: filtered
         data/il_grimmsnarl_v2/pairs.jsonl for the 103 WINNING jiatu.l games
         where the opponent's board (active or bench, any point in the game)
         showed a Crustle-line Pokemon (344/345), decoded every chosen MAIN
         option via cg_native's all_attack() attackId table (see
         scripts/ -- attack table dumped to confirm attackId->name/damage,
         since ATTACK options carry an opaque attackId, not a name). Result:
         across those 103 wins, Munkidori's [Ability] Adrena-Brain fired 1142
         times vs Shadow Bullet's 321 -- roughly 3.5x more often, ~11x/game.
         MECHANISM: Mysterious Rock Inn ("[Ability] ... Prevent all damage
         done to this Pokemon by attacks from your opponent's Pokemon ex")
         blocks Shadow Bullet (Grimmsnarl ex IS a Pokemon ex) when Crustle
         (345) is the opponent's Active -- engine-confirmed independently
         2026-07-21 for a different ex attacker vs this same Ability (see
         project_lopunny_pierce_engine_confirmed memory note). But
         Adrena-Brain is an ABILITY effect ("move ... damage counters"), NOT
         an attack -- it is not blocked by Rock Inn regardless of which
         Pokemon has it or is ex. It also HEALS the source Pokemon by the
         same amount it damages the opponent's target (a strict-upside,
         zero-cost, once-per-turn transfer) -- so a skilled human fires it
         essentially every turn it is available, win or not, not just on
         guaranteed kills. The existing item-3.5 trigger guard only fires on
         a guaranteed-KO-this-turn (far too narrow vs a wall deck this needs
         ~11 turns of grinding to bring down). Two fixes, both gated on a
         board-only tank detector (_opp_has_tank, IDs 344/345/169/190 -- no
         behavior change for any other matchup, so the Alakazam/random win
         rates are structurally untouched):
           (a) _ko_guard bug fix: it was treating Shadow Bullet's listed 180
               damage as always-lethal-eligible, including vs an active
               Crustle (345) where Rock Inn makes the real damage 0 -- a
               guaranteed guard misfire (declares "lethal", takes the
               attack, deals nothing). Now zeroed out specifically for
               Grimmsnarl ex (648) attacking an active Crustle (345).
           (b) _adrena_trigger_guard broadened: in addition to the existing
               guaranteed-KO branches (still used for non-tank matchups,
               e.g. Alakazam, completely unchanged), added an opportunistic
               branch that fires Adrena-Brain whenever `_opp_has_tank(...)`
               is true and there is any damage to siphon (available_snipe >
               0), regardless of lethality -- matching the mined 11x/game
               cadence. Also fixed the same Shadow-Bullet-vs-Crustle-active
               zeroing inside the guard's own KO-combo branch (item 3.5's
               "shadow_bullet_dmg" component), so it no longer assumes a
               blocked attack contributes real damage to a combo kill.
         REJECTED: a Boss's Orders bench-snipe-target guard (explicitly
         asked for in the anti-tank task). Mined it too (78 resolved
         Boss's-Orders plays across the same 103+6 tank-matchup episodes):
         target HP was full or near-full 83% of the time on average (median
         97%) -- i.e. jiatu.l overwhelmingly used it to drag in a *fresh*
         threat (deny setup tempo / force a bad trade), not to snipe an
         already-damaged bench mon for a cheap KO. That is exactly the kind
         of context (MAIN-context Supporter play + its CARD target select)
         the IL ranker already sees plenty of in training (unlike the
         forced-promotion / attack / ability contexts the other guards
         cover) -- and the signal is diluted by non-Crustle high-HP ex
         attackers ("mystery" card 756) present in real ladder decks that
         splash Crustle as a tech, not our synthetic pure-wall v009a/v044/
         v049 gate opponents. Overriding a well-trained ranker on a
         contaminated, already-in-distribution signal risks doing more harm
         than good, so this was left to the ranker (same judgment call
         pattern as the retreat-guard SKIP two paragraphs above).
      4. Deterministic scorer: to_observation_class(obs_dict) -> scoring_grimmsnarl.
         choose_options -> greedy argmax of score_option over the option list.
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

# v061: the LightGBM ranker (v060 step 5) is replaced by a deterministic v034/v036-style
# greedy scorer (scoring_grimmsnarl.choose_options). The guards below still operate on the
# raw obs_dict (integer constants); the scorer operates on the typed obs from
# to_observation_class. Both come from the same engine dict, so option index i is identical
# in both representations.
from cg.api import to_observation_class, all_card_data, all_attack

try:
    import scoring_grimmsnarl as _scorer
    _scorer_ok = True
except Exception as _scorer_err:  # pragma: no cover - defensive
    _scorer = None
    _scorer_ok = False
    print(f"[v061] WARNING: scoring_grimmsnarl import failed: {_scorer_err}", file=sys.stderr)

# ── Instrumentation counters ─────────────────────────────────────────────────
_n_model_decisions: int = 0
_n_ko_guard: int = 0
_n_promo_guard: int = 0
_n_energy_guard: int = 0
_n_adrena_trigger: int = 0   # number of times the Adrena-Brain snipe was initiated (step 1)
_n_adrena_steps: int = 0     # number of forced-continuation steps handled (steps 2-4)
_n_adrena_tank_opportunistic: int = 0  # of the above, how many via the item-3.6 non-lethal tank branch
_n_ko_guard_crustle_blocked: int = 0   # times the item-3.6(a) Rock-Inn zeroing suppressed a false "lethal"
_n_trivial_fallback: int = 0
_exception_count: int = 0

# Debug logging to stderr (Kaggle captures per-move stdout/stderr in agent
# logs). Only used for the new item-3.6 anti-tank branches -- kept quiet
# elsewhere to avoid spamming every decision.
_DEBUG_LOG = True


def _dbg(msg: str) -> None:
    if _DEBUG_LOG:
        try:
            print(f"[dbg] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass

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
FROSLASS_ID = 104        # Stage 1, 90 HP. "Frost Smash" costs {W} -> UNCASTABLE (0 {W} in deck);
                         # Froslass is a PASSIVE engine piece (Freezing Shroud auto-feeds
                         # Adrena-Brain each Checkup), NOT an attacker (v061 fix vs v060 misclass).
SNORUNT_ID = 860         # Basic, 70 HP. "Chilly" costs {W} -> UNCASTABLE; only value = evolve to Froslass.
BASIC_D_ENERGY_ID = 7    # Basic {D} (Darkness) Energy — Shadow Bullet costs {D}{D}.

# ── Anti-tank IDs (added 2026-07-21, see module docstring item 3.6) ─────────
# Crustle line: [Ability] Mysterious Rock Inn -- "Prevent all damage done to
# this Pokemon by attacks from your opponent's Pokemon {ex}." Grimmsnarl ex's
# only attack (Shadow Bullet) IS a Pokemon-ex attack -> 0 damage to an active
# Crustle (engine-confirmed 2026-07-21, see project_lopunny_pierce_engine_
# confirmed memory note -- verified for a different ex attacker vs this same
# Ability, applies identically to Shadow Bullet). Dwebble (pre-evolution) has
# no such Ability -- only card 345 (evolved Crustle) blocks ex-attack damage.
CRUSTLE_ID = 345
DWEBBLE_ID = 344         # Crustle's pre-evolution -- NOT ability-protected.
DURALUDON_ID = 169
ARCHALUDON_ID = 190      # Archaludon ex -- no damage-prevention Ability; Shadow Bullet hits normally.
TANK_LINE_IDS = frozenset({CRUSTLE_ID, DWEBBLE_ID, DURALUDON_ID, ARCHALUDON_ID})

# Adrena-Brain snipe sequence state (module-level; the sub-decisions at steps
# 2-4 are forced continuations of this guard's own step-1 choice, not
# independent MAIN decisions — see module docstring item 3.5).
_adrena_state: str | None = None  # None | "await_from" | "await_count" | "await_target"
# Known fragile/support mons with no meaningful attack (deprioritize as promotion target
# whenever a real attacker is available on bench).
FRAGILE_SUPPORT_IDS = {
    305,   # Dunsparce — 0-dmg switch utility (not in deck; kept as a defensive generic)
    66,    # Dudunsparce — draw-then-shuffle-self utility (not in deck)
    646,   # Marnie's Impidimp — Filch draw / 10-dmg Corkscrew (Grimmsnarl pre-evo, setup only)
    112,   # Munkidori — ability-only (Adrena-Brain); Mind Bend costs {P} -> UNCASTABLE. Worst active.
    860,   # Snorunt — attack UNCASTABLE ({W}); bench fodder that evolves into Froslass.
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

# ── Card DB + damage table (for the lethal-KO guard) ─────────────────────────
# v061 Phase-1 guard edit (c): the lethal-KO guard used to read attack damage from a
# pandas/EN_Card_Data.csv table. Now built straight from the engine via cg.api:
# all_card_data() gives CardData.attacks (a list of attackId ints, ABILITIES ALREADY
# EXCLUDED — those live on CardData.skills), and all_attack() maps attackId -> Attack with
# a .damage field. So _DAMAGE_TABLE[cardId] = [damage per attack, in CardData.attacks
# order], which matches the engine's ATTACK-option ordering the guard indexes by rank.
# Drops three live-failure surfaces (lightgbm, features module, pandas + bundled CSV).
try:
    CARD_DB = {c.cardId: c for c in all_card_data()}
    ALL_ATTACKS = {a.attackId: a for a in all_attack()}

    _DAMAGE_TABLE: dict[int, list[int]] = {}
    for _cid, _cd in CARD_DB.items():
        _dmgs = []
        for _aid in (getattr(_cd, "attacks", None) or []):
            _atk = ALL_ATTACKS.get(_aid)
            _dmgs.append(int(getattr(_atk, "damage", 0) or 0) if _atk else 0)
        _DAMAGE_TABLE[int(_cid)] = _dmgs
    _damage_table_ok = True
except Exception as _dmg_err:
    CARD_DB = {}
    ALL_ATTACKS = {}
    _DAMAGE_TABLE = {}
    _damage_table_ok = False
    print(f"[v061] WARNING: damage table load failed: {_dmg_err}", file=sys.stderr)


# ── Active-promotion guard ───────────────────────────────────────────────────
def _promo_priority(pk_id: int, hp: int) -> tuple[int, int]:
    """(tier, hp) sort key — higher tier wins, hp is the tie-break within a tier.

    v061 Phase-1 guard edit (a): the deck's real attackers are ONLY Grimmsnarl ex (648,
    Shadow Bullet 180) and Morgrem (647, Corkscrew {D}{D} 60). Froslass (104) was
    mis-tiered as a "real attacker" in v060, but Frost Smash costs {W} and there are 0 {W}
    energies in the deck -> it can never attack. It is instead a TANKY UTILITY promotion
    target (90 HP, survives a hit, and its Freezing Shroud keeps feeding Adrena-Brain even
    while Active): strictly better than the fragile/ability-only mons, but below anything
    that can actually threaten damage. Morpeko (649) is not in this deck -> dropped.
    Tiers: 4 Grimmsnarl ex > 3 Morgrem (real attacker) > 2 unlisted > 1 Froslass (tanky
    utility) > 0 fragile/support (Impidimp/Munkidori/Snorunt/…).
    """
    if pk_id == GRIMMSNARL_ID and hp > 0:
        return (4, hp)
    if pk_id == MORGREM_ID and hp > 0:
        return (3, hp)
    if pk_id in FRAGILE_SUPPORT_IDS:
        return (0, hp)
    if pk_id == FROSLASS_ID and hp > 0:
        return (1, hp)
    # Unlisted mon (a tech card we don't model) — better than known-fragile, worse than a
    # real attacker; ranked above Froslass since an unknown mon might actually threaten.
    return (2, hp)


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


# ── Anti-tank board detector (item 3.6) ──────────────────────────────────────
def _opp_has_tank(obs_dict: dict) -> bool:
    """True iff the opponent's board (active or bench, right now) shows any
    Crustle-line (344/345) or Archaludon-line (169/190) Pokemon. Board-only,
    deliberately not stateful/sticky (matches how the other guards read
    obs_dict fresh each call) -- see module docstring item 3.6.
    """
    try:
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        opp_pl = players[1 - me]
        for area in ("active", "bench"):
            for pk in (opp_pl.get(area) or []):
                if isinstance(pk, dict) and int(pk.get("id") or 0) in TANK_LINE_IDS:
                    return True
        return False
    except Exception:
        return False


# ── Prize / weakness helpers (for the game-winning lethal check) ─────────────
def _prize_value(card_id: int) -> int:
    cd = CARD_DB.get(card_id)
    if cd is not None and getattr(cd, "megaEx", False):
        return 3
    if cd is not None and getattr(cd, "ex", False):
        return 2
    return 1


def _weak_to_dark(card_id: int) -> bool:
    """True iff the card's Weakness type is Darkness (== BASIC_D_ENERGY_ID, 7) — our only
    real attacks (Shadow Bullet / Corkscrew) are Dark-type, so weakness doubles their damage."""
    cd = CARD_DB.get(card_id)
    w = getattr(cd, "weakness", None) if cd else None
    if w is None:
        return False
    try:
        return int(getattr(w, "value", w) or 0) == BASIC_D_ENERGY_ID
    except Exception:
        return False


# ── Lethal-KO guard ──────────────────────────────────────────────────────────
def _ko_guard(obs_dict: dict) -> list[int] | None:
    """Take a lethal ATTACK immediately — but ONLY when the KO WINS THE GAME.

    v061 change (legibility probe 2026-07-21): v060 fired the lethal attack the instant any
    ATTACK option's listed damage >= opp active HP. But attacking ENDS the turn, and the
    expert develops the board first (Adrena-Brain / attach energy / play basics) and attacks
    LAST — the low-HP target isn't going anywhere, so the KO still lands this turn after all
    the free setup. The deterministic scorer already sequences attacks last (ATTACK scores
    below development), so it takes non-winning lethals on its own, correctly ordered. This
    guard therefore only FRONT-RUNS the scorer for a GAME-WINNING lethal (the KO takes our
    last needed prize) — never fumbled, taken immediately, matching the expert. Damage is
    weakness-adjusted (Dark-type attacks double vs Dark-weak targets).

    Item 3.6(a) (kept): Grimmsnarl ex (648) vs an active Crustle (345) — Mysterious Rock Inn
    prevents ALL ex-attack damage, so Shadow Bullet's listed 180 is zeroed (no false lethal).
    """
    global _n_ko_guard_crustle_blocked
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
        opp_active_id = int(opp_active.get("id") or 0)
        opp_hp = int(opp_active.get("hp") or 0)
        if opp_hp <= 0:
            return None
        rock_inn_blocks_ex = my_active_id == GRIMMSNARL_ID and opp_active_id == CRUSTLE_ID
        damages = _DAMAGE_TABLE.get(my_active_id) or []
        if not damages:
            return None
        # Game-winning check: does KO'ing the opp Active take our last needed prize(s)?
        my_prize_remaining = len(my_pl.get("prize") or [])
        opp_prize_value = _prize_value(opp_active_id)
        game_winning = my_prize_remaining > 0 and opp_prize_value >= my_prize_remaining
        opp_weak_dark = _weak_to_dark(opp_active_id)

        best_lethal_i = None
        attack_rank = 0
        for i, opt in enumerate(opts):
            if int(opt.get("type") or 0) != OPT_ATTACK:
                continue
            dmg = damages[attack_rank] if attack_rank < len(damages) else (damages[-1] if damages else 0)
            attack_rank += 1
            if rock_inn_blocks_ex:
                if dmg > 0:
                    _n_ko_guard_crustle_blocked += 1
                    _dbg(f"ko_guard: Rock Inn blocks Shadow Bullet vs active Crustle "
                         f"(listed {dmg} dmg -> 0), suppressing false-lethal (opp_hp={opp_hp})")
                dmg = 0  # Mysterious Rock Inn: 0 real damage from Grimmsnarl ex's attack to an active Crustle.
            eff = dmg * 2 if (dmg > 0 and opp_weak_dark) else dmg
            if eff > 0 and eff >= opp_hp:
                best_lethal_i = i
                break
        if best_lethal_i is None:
            return None
        # Only front-run the scorer for a GAME-WINNING lethal. A non-winning KO is sequenced
        # last by the scorer (attacking ends the turn; the low-HP target survives our setup
        # and still dies this turn). Firing every listed lethal immediately mis-sequences vs
        # the expert (legibility probe 2026-07-21: forced_lethal top-1 0.20 -> develop-first).
        if game_winning:
            return [best_lethal_i]
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
    features_grimmsnarl_v2._get_pokemon_from_obs. area: 4=ACTIVE, 5=BENCH."""
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

    Fires in three cases:
      (a) opp active HP <= available snipe damage (direct kill), or
      (b) opp active HP in (shadow_bullet_dmg, shadow_bullet_dmg +
          available_snipe] AND my active Grimmsnarl ex already has its
          {D}{D} Shadow Bullet cost ready (so the hard-lethal-KO guard
          finishes the KO with the attack on the very next MAIN decision,
          once this snipe lands) -- shadow_bullet_dmg is zeroed when the
          target is a Rock-Inn'd active Crustle (item 3.6(a), same fix as
          _ko_guard: that attack does 0 real damage there, so it can't be
          part of a combo kill), or
      (c) item 3.6(b), added 2026-07-21: `_opp_has_tank(obs_dict)` is true
          (opponent board shows Crustle/Archaludon-line, IDs 344/345/169/
          190) and there is ANY damage to siphon (available_snipe > 0) --
          fire opportunistically even without a kill this turn. Mined
          evidence (module docstring item 3.6): jiatu.l's 103 winning games
          vs Crustle-line opponents fired Adrena-Brain ~11x/game, 3.5x more
          than Shadow Bullet -- a wall deck needs sustained per-turn
          grinding, not a one-shot lethal check, and the ability is a
          strict-upside heal-them-damage-you transfer that never carries
          over unused, so there is no reason to hold it back. This branch
          never fires outside a detected tank matchup (Alakazam/random
          behavior is byte-for-byte the old lethal-only logic).
    Available snipe damage = 10 * min(3, max damage-counters on any one of
    my Pokemon) — Adrena-Brain moves "up to 3" counters (30 dmg) from ONE
    of my Pokemon. Sets _adrena_state="await_from" and returns the
    OPT_ABILITY option index on success. See module docstring items 3.5/3.6.
    """
    global _adrena_state, _n_adrena_tank_opportunistic
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
        opp_active_id = int(opp_active.get("id") or 0)
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
                if opp_active_id == CRUSTLE_ID:
                    shadow_bullet_dmg = 0  # item 3.6(a): Rock Inn blocks Shadow Bullet vs active Crustle.
                if shadow_bullet_dmg > 0 and opp_hp > shadow_bullet_dmg and opp_hp <= shadow_bullet_dmg + available_snipe:
                    _adrena_state = "await_from"
                    return [ability_idx]

        # v061 Phase-1 guard edit (b): fire Adrena-Brain OPPORTUNISTICALLY IN EVERY MATCHUP
        # whenever there is damage to siphon (available_snipe > 0 is already guaranteed above —
        # we returned None if max_counters <= 0). v060 gated this on _opp_has_tank (Crustle/
        # Archaludon only); the gate is removed here. Rationale: Froslass's Freezing Shroud puts
        # a damage counter on every ability-mon (both sides, Froslass exempt) each Pokémon
        # Checkup, so there is a free counter to move almost every turn regardless of opponent.
        # The transfer HEALS our source Pokémon and DAMAGES the opponent's Active — a strict-
        # upside, zero-card-cost, once-per-turn effect that never carries over unused. Widening
        # it to all matchups only gains tempo (Alakazam/random included); it can never hurt.
        _adrena_state = "await_from"
        _n_adrena_tank_opportunistic += 1
        _dbg(f"adrena opportunistic fire #{_n_adrena_tank_opportunistic}: "
             f"opp_active_id={opp_active_id} opp_hp={opp_hp} available_snipe={available_snipe} "
             f"tank={_opp_has_tank(obs_dict)}")
        return [ability_idx]
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


# ── Deterministic scorer bridge (replaces v060's LightGBM step 5) ─────────────
def _score_decide(obs_dict: dict) -> list[int]:
    """Bridge the raw-dict guard world to the typed-obs scorer.

    Converts obs_dict -> typed obs via to_observation_class and defers to
    scoring_grimmsnarl.choose_options, which greedily argmax's the deterministic
    score_option over the option list (v034/v036 method). Raises if the scorer is
    unavailable (agent() then falls through to the trivial fallback — never-throw
    preserved). Returns [] for a legitimately-optional decline (minCount 0, no positive
    option), which agent() passes through unchanged.
    """
    if not _scorer_ok or _scorer is None:
        raise RuntimeError("scorer unavailable")
    obs = to_observation_class(obs_dict)
    if not obs.select or not obs.select.option:
        return []
    # Opponent-attack log tracking (defensive: only if the scorer ports it — some Boss
    # heuristics read the opponent's last attack).
    _upd = getattr(_scorer, "_update_opp_attack_tracking", None)
    if _upd is not None:
        try:
            _upd(obs)
        except Exception:
            pass
    return _scorer.choose_options(obs)


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
      2. Lethal-KO guard (any attack that KOs the opponent's active -> take it;
         item 3.6(a) zeroes out Shadow Bullet vs a Rock-Inn'd active Crustle).
      3. Adrena-Brain trigger guard (initiate the Munkidori snipe when it
         converts to a guaranteed KO this turn, direct or combined with this
         turn's Shadow Bullet -- OR, v061 guard edit (b), opportunistically in
         ANY matchup whenever there's any damage to siphon, since Freezing
         Shroud feeds a free counter almost every turn).
      4. Energy-attach guard (active Grimmsnarl ex needs {D}{D} -> attach it).
      5. Deterministic heuristic scorer (scoring_grimmsnarl.choose_options).
      6. Trivial fallback (first minCount indices).
    """
    global _n_model_decisions, _n_ko_guard, _n_promo_guard, _n_energy_guard
    global _n_adrena_trigger, _n_adrena_steps, _n_trivial_fallback, _exception_count
    global _n_adrena_tank_opportunistic, _n_ko_guard_crustle_blocked

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        # New game: reset per-game guard/scorer state.
        global _adrena_state
        _adrena_state = None
        _reset = getattr(_scorer, "reset_game_state", None) if _scorer_ok else None
        if _reset is not None:
            try:
                _reset()
            except Exception:
                pass
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

    # 5. Deterministic heuristic scorer
    try:
        result = _score_decide(obs_dict)
        if (result is not None
                and all(0 <= i < n_options for i in result)
                and min_count <= len(result) <= max_count
                and len(set(result)) == len(result)):
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[v061] scorer exception #{_exception_count}: {exc}  "
                f"(scored={_n_model_decisions} ko={_n_ko_guard} "
                f"fallback={_n_trivial_fallback})",
                file=sys.stderr,
            )

    # 6. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
