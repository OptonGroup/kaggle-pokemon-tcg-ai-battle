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

v062 [форензик] edits on top of v061 (each tagged "[форензик v062]" at its own definition;
see scoring_grimmsnarl.py's module docstring for the scorer-side items 1/2/4/5/6):
  1. reset_game_state() bug fix (scoring_grimmsnarl.py) -- was called here but never defined
     there, so it was a silent no-op and scorer state leaked across games in one process.
  2. _detect_matchup_raw (this file) + detect_matchup (scoring_grimmsnarl.py): sticky
     board-scan matchup detection (crustle/archaludon/lucario/tr_mewtwo/abomasnow/mirror/
     alakazam/generic), ported from v049_bench_sticky_guard's board+sticky layer (no
     torch/belief-net). Gates item 3 below.
  3. _munkidori_energy_guard (NEW, this file, step 4 in the chain below): force the turn's
     {D} energy attach onto a 0-energy Munkidori in grind/tank matchups, so Adrena-Brain
     can actually switch on instead of losing every energy attach to the energy-attach
     guard's greedy claim on Grimmsnarl ex. See that function's own docstring for the full
     diagnosis (3/4 live Crustle losses had Adrena-Brain firing 0 times).

v062 [майнинг v062] wave-2 edits on top of the forensic pass above, from mining 2711 pure
jiatu.l/bono expert games (docs/gold/v062_expert_playbook_lines.md; full aggregates in
data/il_grimmsnarl_v2/expert_playbook.json). Each is tagged "[майнинг v062]" at its own
definition:
  1. _munkidori_energy_guard: the matchup gate is REMOVED (was
     `_MUNKIDORI_ENERGY_GATE_MATCHUPS == {crustle,archaludon,lucario,tr_mewtwo,abomasnow}`).
     Playbook lines 14-19: the expert sends the game's FIRST energy to Munkidori in EVERY
     matchup bucket (48-80% of ordinal-1 attaches, median turn 1-2 everywhere -- alakazam/
     mirror/generic included, not just grind matchups), and Munkidori gets *an* energy at
     all in 90-100% of games in every bucket. Lines 55-57/169-177: the single starkest
     won/lost Adrena-Brain-activation-volume gap in the whole dataset is actually vs
     Alakazam (4.21W vs 2.15L activations/game, ~2x) -- exactly the matchup the wave-1 gate
     excluded. The guard now fires in every matchup; its own 0-energy self-limit (only
     matches a Munkidori with 0 {D} already attached) is the only remaining restriction.
  2. _adrena_sequence_guard's "await_target" step (ctx CTX_DAMAGE_COUNTER): the
     opportunistic ("value") trigger branch now routes to the opponent's BENCH by default
     instead of the active. Playbook lines 58-61 (Alakazam: Adrena-Brain targets are
     65.4%/79.5% opponent-bench in won/lost games) and lines 124-149 (Archaludon
     reconstruction: damage is routed onto the opponent's not-yet-evolved backup Duraludon
     copies sitting on bench -- "kill the reinforcement before it evolves") both point the
     same direction across matchups. The guaranteed-KO ("kill") and KO-combo ("combo")
     trigger branches are UNCHANGED -- still target the active, since that target was
     already fixed when the guard chose to fire on those paths. A new module-level
     `_adrena_intent` global (set by `_adrena_trigger_guard`, consumed by
     `_adrena_sequence_guard`, reset alongside `_adrena_state`) carries which branch
     initiated the current combo through steps 2-4.
  3. Boss's Orders matchup-specific target-priority tables (BOSS_TARGET_PRIORITY,
     `_boss_tier_bonus`) live in scoring_grimmsnarl.py -- see that module's docstring item 3.
  4. score_retreat (scoring_grimmsnarl.py) is UNCHANGED by this pass -- verified against
     playbook lines 20-24 (median HP% at retreat == 100.0 in every one of the 8 matchup x
     outcome buckets): the wave-1 tempo-scheduled Munkidori retreat (promote a real
     attacker off the bench, not an HP-triggered flee) already matches expert play. No
     speculative HP-threshold retreat guard was added, per the mined evidence against it.

v074 P0 -- STADIUM-AWARE DAMAGE in the raw-dict guards (ТЗ docs/gold/PLAN_v074.md; the
scorer half is documented in scoring_grimmsnarl.py's docstring). Each tagged "[ТЗ v074 P0]":
  * `_stadium_damage_reduction(cur, card_id)` -- raw-dict twin of the scorer's
    `stadium_damage_reduction`: 30 iff Full Metal Lab (1244) is the stadium in play and the
    target is a {M} Pokemon. Both halves of the pilot must agree on one damage number;
    v073's smoking gun was a single decision where the guard and the scorer disagreed.
  * `_ko_guard`: subtracts it AFTER weakness (card text order), so the guard no longer
    front-runs the scorer to take a "game-winning lethal" that lands 30 short vs a {M} body.
  * `_adrena_trigger_guard` KO-combo branch: Shadow Bullet is 150, not 180, under the
    stadium -> the "snipe now, KO next" window is 30 narrower and the once-per-turn ability
    is not spent on a combo that cannot close. The DIRECT-kill branch is deliberately
    untouched: Adrena-Brain MOVES damage counters, it is not damage from an attack, so the
    stadium does not apply to it.
  * `_promotion_guard` is deliberately untouched by the FULL METAL LAB half: its premise
    ("a charged ex swings 180 before dying") stays TRUE under the stadium (150 > 0) --
    unlike the Rock-Inn case in v073, where the swing was actually zero.

v074c P1 -- ROCK-INN DEMOTE UNGATED (форензик v073, out/v073_full_download/ANALYSIS_v073.md
§7.1). The v073 headline fix was measured at +3.85pp on the crustle wall, but its
`_promotion_guard` half never ran in the scenario it was written for: the whole `dies_free`
block sat behind `_opp_threat_vs(...) >= hp`, i.e. "the opponent kills this ex right now".
Crustle 345 is a low-damage WALL, so a healthy ex never satisfied that, kept its full
promotion key, and drove into the Active spot to deal ZERO through Mysterious Rock Inn.
Live proof, 87924206 (crustle, LOSS), twice in one game: the guard demoted a 60hp copy and
then PICKED the second ex at 290hp over Munkidori(60)/Froslass(90) fodder.
Fix: `ex_blocked_by_wall` is now an independent reason to yield the Active spot, no longer
conjoined with `doomed`. The v072 rule for the OTHER reason is untouched (`doomed and
d_en < 2`), because demoting a CHARGED doomed ex cost -5.7pp vs ext_nursrijan_lucario on an
N=1000 gate. New `[dbg]` tag "[v074c P1: healthy ex, v073 gate missed it]" + counter
`_n_promo_rockinn_healthy` make the added firings countable in a gate log.
"""
from __future__ import annotations

import hashlib
import json
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
_n_munkidori_energy_guard: int = 0  # [форензик v062] item 3: energy forced onto a 0-{D} Munkidori
_n_adrena_trigger: int = 0   # number of times the Adrena-Brain snipe was initiated (step 1)
_n_adrena_steps: int = 0     # number of forced-continuation steps handled (steps 2-4)
_n_adrena_tank_opportunistic: int = 0  # of the above, how many via the item-3.6 non-lethal tank branch
_n_ko_guard_crustle_blocked: int = 0   # times the item-3.6(a) Rock-Inn zeroing suppressed a false "lethal"
_n_promo_rockinn_demote: int = 0       # [v073 P0] charged-but-blocked ex yielded the Active spot to fodder
_n_promo_rockinn_healthy: int = 0      # [форензик v073 -> v074c P1] of the above, the ones the OLD
                                       # `threat >= hp` gate silently skipped (healthy ex vs the wall)
_n_ko_guard_fml_blocked: int = 0       # [ТЗ v074 P0] times Full Metal Lab's -30 killed a false "lethal"
_n_adrena_fml_adjusted: int = 0        # [ТЗ v074 P0] times the Adrena combo used the reduced Shadow Bullet
_n_trivial_fallback: int = 0
_exception_count: int = 0
_n_loop_breaker: int = 0

# [форензик v063 ДЕФЕКТ-6] loop-breaker state: md5 of the last (select, current) pair and
# how many consecutive times it repeated verbatim. Live v062 lost episodes to the engine
# re-presenting an identical select until the step limit (prize-take loop 87341448, stalled
# selection 87339069, switch loop 87355692) while the deterministic agent kept answering
# identically. Reset on every new-game boundary in agent().
_loop_sig: str | None = None
_loop_repeat: int = 0

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


def _galt_summary(opts) -> str:
    """Compact 'what was on the table' dump of a raw select.option list, for
    the [galt] guard-alternatives trace. Reads only the raw dict fields
    (type/cardId) per the Option dataclass in cg/api.py -- cheap, no
    to_observation_class needed.
    """
    parts = []
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        t_s = str(int(t)) if t is not None else "?"
        cid = o.get("cardId")
        c_s = str(int(cid)) if cid is not None else "None"
        parts.append(f"#{i} t{t_s} c{c_s}")
    return "; ".join(parts)


def _dbg_alt(name: str, opts) -> None:
    # [форензик v062] companion to _dbg: when a guard forces a decision, dump
    # the raw options it chose among (index/type/cardId) so a downloaded
    # Kaggle log shows what the guard passed over, not just what it picked.
    if _DEBUG_LOG:
        try:
            print(f"[galt] guard={name} opts: {_galt_summary(opts)}", file=sys.stderr, flush=True)
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
ARCHALUDON_ID = 190      # Archaludon ex -- no damage-prevention Ability, but see Full Metal Lab
                         # below: under it Shadow Bullet lands 150, not the listed 180.
TANK_LINE_IDS = frozenset({CRUSTLE_ID, DWEBBLE_ID, DURALUDON_ID, ARCHALUDON_ID})

# ── [ТЗ v074 P0] Full Metal Lab (Stadium 1244) ───────────────────────────────
# "{M} Pokemon (both yours and your opponent's) take 30 less damage from attacks from the
# opponent's Pokemon (after applying Weakness and Resistance)". Every archaludon deck in
# the corpus (and v044/v036 in the gate pool) runs 3-4 copies; our own deck has no {M}
# Pokemon, so it is a one-directional tax on OUR damage. Engine-measured, 120 games:
# Shadow Bullet into a {M} Active = 150 under it (180 never occurs), and the 30 bench
# splash lands ZERO on a {M} bencher. The scorer routes this through
# scoring_grimmsnarl.stadium_damage_reduction; the raw-dict guards below need their own
# copy for the same reason the Rock-Inn rule has one (they run BEFORE the scorer, on the
# raw obs_dict, and must agree with it -- v073's smoking gun was exactly two disagreeing
# damage estimates for one decision).
FULL_METAL_LAB_ID = 1244
METAL_ENERGY = 8         # EnergyType.METAL

# ── [ТЗ v074 P0] Premium Power Pro (Item 1141) ────────────────────────────────
# The INCOMING mirror of Full Metal Lab: "+30 damage from your {F} Pokemon to your
# opponent's Active this turn". Observable in `PlayerState.discard`, which the engine
# exposes for BOTH players. The estimator itself lives in scoring_grimmsnarl
# (`ppp_threat_bonus`) and takes plain scalars precisely so this raw-dict guard and the
# typed-obs scorer produce ONE number for one decision -- v073's §7.1 defect was two halves
# of the pilot disagreeing about damage.
PREMIUM_POWER_PRO_ID = 1141

# ── Matchup-line IDs for _detect_matchup_raw (added v062, item 2) ────────────
# Same priority-ordered line set as scoring_grimmsnarl.py's typed-obs detect_matchup
# (kept in sync deliberately -- two representations of the same engine dict, see module
# docstring's "Architecture split"). Grind/tank matchups the item-3 Munkidori energy
# guard gates on.
CRUSTLE_LINE_IDS = frozenset({CRUSTLE_ID, DWEBBLE_ID})          # 345, 344
ARCHALUDON_LINE_IDS = frozenset({DURALUDON_ID, ARCHALUDON_ID})  # 169, 190
LUCARIO_LINE_IDS = frozenset({677, 678})
TR_MEWTWO_LINE_IDS = frozenset({431})
ABOMASNOW_LINE_IDS = frozenset({722, 723})
STARMIE_LINE_IDS = frozenset({1031, 1030})  # [форензик v063 92-эп live] item 1 -- Mega Starmie
                                             # ex heal-wall (Wally's Compassion) + Staryu bench
MIRROR_LINE_IDS = frozenset({646, 647, 648})  # opponent board showing our own Marnie IDs
ALAKAZAM_LINE_IDS = frozenset({741, 742, 743})

# [форензик v064 БАГ-9] hand-scaling attacks: listed damage == "n/a"/0 in the engine's
# damage table, but the real damage scales with a hand size at attack time. Alakazam
# (743)'s Powerful Hand: 2 damage counters (20 HP) per card in the ATTACKING PLAYER's
# hand (live replay 87420477 -- 620 damage with a 31-card hand). Key = attacker card-id,
# value = (HP dealt per card in hand, minimum energy required) -- mirrors
# scoring_grimmsnarl.py's SCALING_ATTACKERS (kept as a separate raw-dict constant here,
# same "two representations of the same engine dict" split as _detect_matchup_raw above).
_SCALING_ATTACKERS = {743: (20, 1)}

_MATCHUP_PRIORITY_RAW = (
    (CRUSTLE_LINE_IDS, "crustle"),
    (ARCHALUDON_LINE_IDS, "archaludon"),
    (LUCARIO_LINE_IDS, "lucario"),
    (TR_MEWTWO_LINE_IDS, "tr_mewtwo"),
    (ABOMASNOW_LINE_IDS, "abomasnow"),
    (STARMIE_LINE_IDS, "starmie"),
    (MIRROR_LINE_IDS, "mirror"),
    (ALAKAZAM_LINE_IDS, "alakazam"),
)
# [майнинг v062] REMOVED: `_MUNKIDORI_ENERGY_GATE_MATCHUPS`, the wave-1 gate that limited
# the item-3 Munkidori energy guard to {crustle, archaludon, lucario, tr_mewtwo, abomasnow}.
# docs/gold/v062_expert_playbook_lines.md lines 14-19 (mined from 2711 jiatu.l/bono expert
# games): the game's FIRST energy goes to Munkidori in EVERY matchup bucket (48-80% of
# ordinal-1 attaches, median turn 1-2 everywhere, alakazam/mirror/generic included -- not
# just the grind matchups this gate assumed), and the strongest win/loss correlate in the
# whole dataset (Adrena-Brain activation volume, lines 55-57) is actually STARKEST vs
# Alakazam (4.21W vs 2.15L/game) -- precisely the matchup this gate excluded. See module
# docstring's wave-2 item 1 for the full citation.

_raw_last_matchup: str | None = None  # sticky global, reset in agent() on new-game boundary

# ── Matchup-line IDs for _detect_matchup_raw (added v062, item 2) ────────────
# Same priority-ordered line set as scoring_grimmsnarl.py's typed-obs detect_matchup
# (kept in sync deliberately -- two representations of the same engine dict, see module
# docstring's "Architecture split"). Grind/tank matchups the item-3 Munkidori energy
# guard gates on.
CRUSTLE_LINE_IDS = frozenset({CRUSTLE_ID, DWEBBLE_ID})          # 345, 344
ARCHALUDON_LINE_IDS = frozenset({DURALUDON_ID, ARCHALUDON_ID})  # 169, 190
LUCARIO_LINE_IDS = frozenset({677, 678})
TR_MEWTWO_LINE_IDS = frozenset({431})
ABOMASNOW_LINE_IDS = frozenset({722, 723})
STARMIE_LINE_IDS = frozenset({1031, 1030})  # [форензик v063 92-эп live] item 1 -- Mega Starmie
                                             # ex heal-wall (Wally's Compassion) + Staryu bench
MIRROR_LINE_IDS = frozenset({646, 647, 648})  # opponent board showing our own Marnie IDs
ALAKAZAM_LINE_IDS = frozenset({741, 742, 743})

# [форензик v064 БАГ-9] hand-scaling attacks: listed damage == "n/a"/0 in the engine's
# damage table, but the real damage scales with a hand size at attack time. Alakazam
# (743)'s Powerful Hand: 2 damage counters (20 HP) per card in the ATTACKING PLAYER's
# hand (live replay 87420477 -- 620 damage with a 31-card hand). Key = attacker card-id,
# value = (HP dealt per card in hand, minimum energy required) -- mirrors
# scoring_grimmsnarl.py's SCALING_ATTACKERS (kept as a separate raw-dict constant here,
# same "two representations of the same engine dict" split as _detect_matchup_raw above).
_SCALING_ATTACKERS = {743: (20, 1)}

_MATCHUP_PRIORITY_RAW = (
    (CRUSTLE_LINE_IDS, "crustle"),
    (ARCHALUDON_LINE_IDS, "archaludon"),
    (LUCARIO_LINE_IDS, "lucario"),
    (TR_MEWTWO_LINE_IDS, "tr_mewtwo"),
    (ABOMASNOW_LINE_IDS, "abomasnow"),
    (STARMIE_LINE_IDS, "starmie"),
    (MIRROR_LINE_IDS, "mirror"),
    (ALAKAZAM_LINE_IDS, "alakazam"),
)
# [майнинг v062] REMOVED: `_MUNKIDORI_ENERGY_GATE_MATCHUPS`, the wave-1 gate that limited
# the item-3 Munkidori energy guard to {crustle, archaludon, lucario, tr_mewtwo, abomasnow}.
# docs/gold/v062_expert_playbook_lines.md lines 14-19 (mined from 2711 jiatu.l/bono expert
# games): the game's FIRST energy goes to Munkidori in EVERY matchup bucket (48-80% of
# ordinal-1 attaches, median turn 1-2 everywhere, alakazam/mirror/generic included -- not
# just the grind matchups this gate assumed), and the strongest win/loss correlate in the
# whole dataset (Adrena-Brain activation volume, lines 55-57) is actually STARKEST vs
# Alakazam (4.21W vs 2.15L/game) -- precisely the matchup this gate excluded. See module
# docstring's wave-2 item 1 for the full citation.

_raw_last_matchup: str | None = None  # sticky global, reset in agent() on new-game boundary


def _detect_matchup_raw(obs_dict: dict) -> str:
    """Raw-dict extension of _opp_has_tank (above): same board-scan-of-IDs approach, widened
    to the full priority-ordered line set (crustle > archaludon > lucario > tr_mewtwo >
    abomasnow > starmie > mirror > alakazam > generic) and made STICKY across a single game --
    once the board confidently shows a line, keep that label even on turns where the identifying
    Pokemon isn't currently visible (KO'd, or simply not re-scanned this turn), since the
    opponent's deck archetype is fixed for the whole game. Mirrors scoring_grimmsnarl.py's
    typed-obs detect_matchup/its sticky pattern (ported from v049_bench_sticky_guard) --
    kept as a SEPARATE raw-dict implementation here (not shared code) since main.py's guards
    intentionally never import from the typed-obs scorer module (see module docstring
    "Architecture split": guards run on the raw dict, before to_observation_class is even
    called for this decision). Never-throw: any failure falls back to the last sticky value,
    or "generic" if none yet.
    """
    global _raw_last_matchup
    try:
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        opp_pl = players[1 - me]
        ids = set()
        for area in ("active", "bench"):
            for pk in (opp_pl.get(area) or []):
                if isinstance(pk, dict):
                    ids.add(int(pk.get("id") or 0))
        label = None
        for line_ids, name in _MATCHUP_PRIORITY_RAW:
            if ids & line_ids:
                label = name
                break
        if label is not None:
            _raw_last_matchup = label
            return label
        return _raw_last_matchup or "generic"
    except Exception:
        return _raw_last_matchup or "generic"

# Adrena-Brain snipe sequence state (module-level; the sub-decisions at steps
# 2-4 are forced continuations of this guard's own step-1 choice, not
# independent MAIN decisions — see module docstring item 3.5).
_adrena_state: str | None = None  # None | "await_from" | "await_count" | "await_target"
# [майнинг v062] item 2 -- which _adrena_trigger_guard branch initiated the pending combo:
# "kill" (direct-KO branch a) | "combo" (Shadow-Bullet-combo branch b) | "value" (opportunistic
# branch c). Consumed by _adrena_sequence_guard's "await_target" step to decide active
# (kill/combo, unchanged from wave 1) vs opponent-bench (value, new in wave 2) targeting.
# Reset to None everywhere `_adrena_state` is reset to None.
_adrena_intent: str | None = None
# [майнинг v062] item 2 -- damage-counter count (0-3) chosen in the "await_count" step,
# needed by "await_target" to know how much HP (count*10) a bench candidate transfer can
# finish off. Only meaningful while _adrena_state in ("await_target",); stale otherwise.
_adrena_move_count: int = 0
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
def _weak_to_type(card_id: int, energy_type) -> bool:
    """[форензик v063] True iff the card's Weakness matches the given EnergyType value
    (generalizes _weak_to_dark for estimating the OPPONENT's attack damage on us)."""
    if energy_type is None:
        return False
    cd = CARD_DB.get(card_id)
    w = getattr(cd, "weakness", None) if cd else None
    if w is None:
        return False
    try:
        return int(getattr(w, "value", w) or 0) == int(getattr(energy_type, "value", energy_type) or 0)
    except Exception:
        return False


def _opp_threat_vs(opp_active_id: int, cand_id: int, opp_hand: int = 0,
                   ppp_bonus: int = 0) -> int:
    """[форензик v063 ДЕФЕКТ-4] Max listed attack damage of the opponent's Active vs a
    candidate of ours, weakness-adjusted. [форензик v064 БАГ-9] `opp_hand` (optional,
    defaults to 0 for existing call sites that don't have it handy) feeds
    _SCALING_ATTACKERS for hand-scaling attacks (e.g. Alakazam 743's Powerful Hand) whose
    listed damage is 0/"n/a" and would otherwise be silently read as no threat at all.
    [ТЗ v074 P0] `ppp_bonus` is Premium Power Pro's observed contribution, added BEFORE the
    weakness doubling per card text -- computed once per decision by the caller so that this
    guard and the scorer's `opp_max_attack_damage` agree on one threat number."""
    best = 0
    if _damage_table_ok:
        dmgs = _DAMAGE_TABLE.get(opp_active_id) or []
        best = max(dmgs) if dmgs else 0

    # [форензик v064 БАГ-9] hand-scaling (Powerful Hand): listed=0, реальный урон = per_card*opp_hand
    sc = _SCALING_ATTACKERS.get(opp_active_id)
    if sc is not None:
        per_card, _min_e = sc
        best = max(best, per_card * int(opp_hand or 0))

    if best <= 0:
        return 0
    best += max(0, int(ppp_bonus or 0))          # [ТЗ v074 P0] before Weakness, per card text
    opp_cd = CARD_DB.get(opp_active_id)
    if _weak_to_type(cand_id, getattr(opp_cd, "energyType", None) if opp_cd else None):
        best *= 2
    return best


def _promo_priority(pk: dict) -> tuple[int, int, int]:
    """(tier, ready, hp) sort key — higher tier wins, attack-readiness then hp break ties.

    v061 Phase-1 guard edit (a): the deck's real attackers are ONLY Grimmsnarl ex (648,
    Shadow Bullet 180) and Morgrem (647, Corkscrew {D}{D} 60). Froslass (104) was
    mis-tiered as a "real attacker" in v060, but Frost Smash costs {W} and there are 0 {W}
    energies in the deck -> it can never attack. It is instead a TANKY UTILITY promotion
    target (90 HP, survives a hit, and its Freezing Shroud keeps feeding Adrena-Brain even
    while Active): strictly better than the fragile/ability-only mons, but below anything
    that can actually threaten damage. Morpeko (649) is not in this deck -> dropped.
    Tiers: 4 Grimmsnarl ex > 3 Morgrem (real attacker) > 2 unlisted > 1 Froslass (tanky
    utility) > 0 fragile/support (Impidimp/Munkidori/Snorunt/…).

    [форензик v063 БАГ-2] `ready` (has its {D}{D} Shadow Bullet/Corkscrew cost attached)
    now breaks ties BEFORE hp: live 87346272 promoted a 0-energy Grimmsnarl ex over the
    fully-charged copy sitting next to it (old (tier, hp) key preferred the healthy-but-
    empty body), leaving us unable to attack OR retreat for 3 turns.
    """
    pk_id = int(pk.get("id") or 0)
    hp = int(pk.get("hp") or 0)
    d_en = sum(1 for e in (pk.get("energies") or []) if int(e or 0) == BASIC_D_ENERGY_ID)
    ready = 1 if (pk_id in (GRIMMSNARL_ID, MORGREM_ID) and d_en >= 2) else 0
    if pk_id == GRIMMSNARL_ID and hp > 0:
        return (4, ready, hp)
    if pk_id == MORGREM_ID and hp > 0:
        return (3, ready, hp)
    if pk_id in FRAGILE_SUPPORT_IDS:
        return (0, ready, hp)
    if pk_id == FROSLASS_ID and hp > 0:
        return (1, ready, hp)
    # Unlisted mon (a tech card we don't model) — better than known-fragile, worse than a
    # real attacker; ranked above Froslass since an unknown mon might actually threaten.
    return (2, ready, hp)


def _promotion_guard(obs_dict: dict) -> list[int] | None:
    """Forced-promotion / switch guard (ctx TO_ACTIVE / SWITCH).

    Bypasses the IL ranker entirely for this context (OOD — ranker trained
    only on MAIN-context decisions; see module docstring for trace evidence).
    Deterministically picks the best available bench Pokemon:
    Grimmsnarl ex (healthy) > other real attacker > unlisted mon > fragile/support.
    Returns None if the option list doesn't look like the expected bench-CARD
    shape (falls through to IL / trivial fallback — never-throw preserved).
    """
    global _n_promo_rockinn_demote, _n_promo_rockinn_healthy
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

        # [форензик v063 ДЕФЕКТ-4] endgame prize-denial: if promoting a candidate whose KO
        # both HANDS THE GAME (opp needs <= its prize value) and is within the opponent's
        # expected one-hit damage, rank it below every non-losing candidate. Only demotes
        # when a non-losing alternative exists (tuple ordering guarantees that).
        opp_pl = players[1 - me]
        opp_needs = len(opp_pl.get("prize") or [])
        opp_active = (opp_pl.get("active") or [None])[0]
        opp_active_id = int(opp_active.get("id") or 0) if isinstance(opp_active, dict) else 0

        opp_hand = int(opp_pl.get("handCount") or 0)

        # [ТЗ v074 P0] one Premium Power Pro estimate for this whole decision, computed from
        # the raw dict and routed through the SAME scorer helper the typed-obs half uses.
        opp_ppp_seen = sum(1 for c in (opp_pl.get("discard") or [])
                           if isinstance(c, dict)
                           and int(c.get("id") or 0) == PREMIUM_POWER_PRO_ID)
        ppp_bonus = 0
        if _scorer_ok and opp_ppp_seen > 0 and opp_active_id > 0:
            try:
                ppp_bonus = int(_scorer.ppp_threat_bonus(
                    opp_active_id, opp_ppp_seen, opp_hand, int(cur.get("turn") or 0)) or 0)
            except Exception:
                ppp_bonus = 0

        best_i = -1
        best_key = None
        cand_log = []
        for i, o in enumerate(opts):
            if int(o.get("type") or -1) != OPT_CARD:
                continue
            if int(o.get("area") if o.get("area") is not None else -1) != 5:  # AreaType.BENCH
                continue
            opt_player = o.get("playerIndex")
            if opt_player is not None and int(opt_player) != me:
                continue
            # [форензик v063 БАГ-1] index 0 is a VALID bench slot: `int(o.get("index") or -1)`
            # collapsed it to -1 (Python falsy 0), silently skipping bench slot 0 -- live
            # 87387027/87346991/87350515 promoted a fragile Munkidori over the healthy
            # Grimmsnarl ex sitting in slot 0. Reproduced deterministically before the fix.
            idx = int(o.get("index") if o.get("index") is not None else -1)
            if not (0 <= idx < len(my_bench)):
                continue
            pk = my_bench[idx]
            if not isinstance(pk, dict):
                continue
            pk_id = int(pk.get("id") or 0)
            hp = int(pk.get("hp") or 0)
            hands_game = (
                opp_needs > 0
                and _prize_value(pk_id) >= opp_needs
                and opp_active_id > 0
                and hp > 0
                # [форензик v064 БАГ-9] pass opp_hand so a hand-scaling attacker (Alakazam
                # 743) is no longer read as 0 threat here -- this is exactly the guard that
                # was dead against it at match point (see module docstring / PLAN v064).
                and _opp_threat_vs(opp_active_id, pk_id, opp_hand, ppp_bonus) >= hp
            )
            # [v072/M2+M1, разведка топ-5 07-24] sacrificial promotion discipline, FORCED
            # promotion (CTX_TO_ACTIVE) only: a 2-prize attacker that the opponent's Active
            # one-shots (threat >= hp) ranks BELOW every cheaper body unless its own attack
            # lethals the opp Active right now (charged + 180/360 >= opp hp). Experts
            # (szlachetny snieg / jiatu.l) promote Impidimp/Munkidori after every KO and let
            # the ex charge on the bench; reactively feeding fresh ex/fragile bodies into
            # repeated OHKOs is a documented top-pilot loss pattern (Taichicchi ep87610636)
            # and cost us 87532276 (charged ex promoted, eaten for 360, -2 prizes).
            # Voluntary SWITCH keeps the old key: score_retreat's defensive-swap plan
            # expects the charged fresh copy to come in (retreat-then-attack), not fodder.
            dies_free = 0
            if (_prize_value(pk_id) >= 2 and hp > 0 and opp_active_id > 0
                    and (ctx == CTX_TO_ACTIVE or opp_active_id == CRUSTLE_ID)):
                d_en = sum(1 for e in (pk.get("energies") or [])
                           if int(e or 0) == BASIC_D_ENERGY_ID)
                # [форензик v073 §7.1 -> v074c P1] `doomed` USED TO BE the entry condition of
                # this whole block, which is what made v073's headline fix a no-op in its own
                # main scenario. Crustle 345 is a WALL: its listed damage is small, so a
                # healthy Grimmsnarl ex never satisfies `threat >= hp`, never got its key
                # demoted, and drove into the Active spot to swing ZERO. Live proof
                # (87924206, crustle, LOSS): twice in one game the guard demoted a 60hp copy
                # while PICKING the second ex at 290hp over Munkidori/Froslass fodder.
                # The two reasons to yield the Active spot are now independent:
                #   * `ex_blocked_by_wall` -- our ex deals 0 through Mysterious Rock Inn, so a
                #     2-prize body in the Active spot is a pure donation REGARDLESS of whether
                #     it dies this turn. Non-ex fodder is NOT blocked by Rock Inn, so it both
                #     deals real damage and costs one prize instead of two.
                #   * `doomed and d_en < 2` -- the v072 rule, unchanged: an UNREADY ex about
                #     to be one-shot dies having done nothing, so fodder goes first. A CHARGED
                #     doomed ex still gets promoted (it swings 180 before dying) -- removing
                #     that cost -5.7pp vs ext_nursrijan_lucario on an N=1000 gate.
                # [v074f P1-fix-a] The `doomed and d_en < 2` rule below stays FORCED-promotion
                # only (it is the v072 sacrificial-promotion discipline, and voluntary switches
                # are planned by score_retreat as retreat-then-attack). The WALL rule is not:
                # against an Active Crustle the ex swings zero no matter how it got there, so
                # exempting CTX_SWITCH just handed the demote straight back. Measured on the
                # v074c gate log: 79 of 114 voluntary switches promoted 648 into the wall, one
                # turn after the forced promotion had demoted it -- net effect of P1 in that
                # subset was "burn one {D} on the retreat", which is why the crustle stratum
                # came out flat.
                doomed = (ctx == CTX_TO_ACTIVE
                          and _opp_threat_vs(opp_active_id, pk_id, opp_hand, ppp_bonus) >= hp)
                # [v072 gate-1 fix] Only an UNREADY doomed ex yields to fodder. The first
                # version demoted CHARGED doomed exes too -- the promoted fodder Impidimp
                # is a MARNIE_LINE "attacker", so score_retreat's unstick branch never
                # rotates it out: it sat Active spamming 10-dmg Filch while the charged
                # Grimmsnarl rotted benched (gate N=1000: ext_nursrijan_lucario 30.8%->
                # 25.1%, p=0.0045). A charged ex swings 180 before dying -- promote it
                # (v065 behavior); an empty ex dies having done nothing -- fodder first.
                #
                # [форензик v072 НАХОДКА 1 -> v073 P0] "A charged ex swings 180 before
                # dying" is FALSE against an active Crustle 345: Mysterious Rock Inn
                # prevents all damage from our ex, so the promoted ex swings ZERO and then
                # hands over 2 prizes. The premise, not the charge state, is what makes a
                # doomed ex worth promoting -- so gate on "can it actually deal damage",
                # which is exactly the rule _ko_guard (main.py:821-843) and the scorer's
                # ATTACK branch already apply. Live cost: 6 games, up to 4 repeats in one
                # (87845558 promoted the charged ex into the wall four separate times;
                # 87857782 twice in a row; also 87890163/87880517/87839149/87846092).
                ex_blocked_by_wall = (pk_id == GRIMMSNARL_ID and opp_active_id == CRUSTLE_ID)
                if ex_blocked_by_wall:
                    dies_free = 1
                    _n_promo_rockinn_demote += 1
                    if not doomed:
                        _n_promo_rockinn_healthy += 1
                    _dbg(f"promo: Rock-Inn demote #{_n_promo_rockinn_demote} "
                         f"{'charged' if d_en >= 2 else 'unready'} ex (d_en={d_en}, hp={hp}, "
                         f"doomed={int(doomed)}) deals 0 vs active Crustle, yielding the "
                         f"Active spot to fodder"
                         + ("" if doomed else "  [v074c P1: healthy ex, v073 gate missed it]"))
                elif doomed and d_en < 2:
                    dies_free = 1
            key = (0 if hands_game else 1, 1 - dies_free) + _promo_priority(pk)
            cand_log.append((i, pk_id, key))
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        if best_i >= 0:
            # [форензик v064 БАГ-10] diagnostics: dump every candidate's (hands_game, tier,
            # ready, hp) key alongside the chosen index, so a downloaded live trace can show
            # WHY this bench mon was promoted instead of another (candidates are
            # (option_index, card_id, key) triples; key = (hands_game_flag, tier, ready, hp)).
            _dbg(f"guard=promotion choice i={best_i} key={best_key} n_cands={len(cand_log)} "
                 f"cands={cand_log}")
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


def _stadium_damage_reduction(cur: dict, card_id: int) -> int:
    """[ТЗ v074 P0] raw-dict twin of scoring_grimmsnarl.stadium_damage_reduction: 30 iff
    Full Metal Lab (1244) is the stadium in play and `card_id` is a {M} Pokemon, else 0.
    `cur` is obs_dict["current"]; its "stadium" key is the single global stadium slot
    (list of 0 or 1 cards -- cg/api.py State.stadium)."""
    try:
        stadium = cur.get("stadium") or []
        if not any(isinstance(c, dict) and int(c.get("id") or 0) == FULL_METAL_LAB_ID
                   for c in stadium):
            return 0
        cd = CARD_DB.get(int(card_id or 0))
        et = getattr(cd, "energyType", None) if cd else None
        if et is None:
            return 0
        return 30 if int(getattr(et, "value", et)) == METAL_ENERGY else 0
    except Exception:
        return 0


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
    global _n_ko_guard_crustle_blocked, _n_ko_guard_fml_blocked
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
        fml_cut = _stadium_damage_reduction(cur, opp_active_id)   # [ТЗ v074 P0]
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
            # [ТЗ v074 P0] Full Metal Lab is applied by the engine AFTER Weakness/Resistance
            # (card text), so subtract it here, not from the listed damage. Without this the
            # guard front-runs the scorer for a "game-winning lethal" that lands 30 short
            # against every {M} body -- we end the turn on an attack that doesn't win instead
            # of developing the board first.
            if eff > 0 and fml_cut > 0:
                eff = max(0, eff - fml_cut)
                if eff < opp_hp:
                    _n_ko_guard_fml_blocked += 1
                    _dbg(f"ko_guard: Full Metal Lab cuts 30 vs {{M}} active id={opp_active_id} "
                         f"({eff + fml_cut} -> {eff}), suppressing false-lethal (opp_hp={opp_hp})")
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


# ── Munkidori energy guard (item 3, [форензик v062] + [майнинг v062] item 1) ────────────
def _munkidori_energy_guard(obs_dict: dict) -> list[int] | None:
    """[форензик v062] item 3, gate REMOVED by [майнинг v062] item 1 -- MAIN-context guard:
    force-attach a Basic {D} Energy from hand onto a 0-energy Munkidori (112), active OR
    bench, in ANY matchup (see "Gate" paragraph below -- was restricted to grind/tank
    matchups in the wave-1 forensic pass, now universal per the mined evidence). Runs
    BETWEEN _adrena_trigger_guard and _energy_guard in the agent() chain, so Munkidori gets
    first claim on the turn's one manual energy attach.

    DIAGNOSIS: forensic review of v061's live losses vs Crustle (memory note
    project_v061_v050_full_forensic_20260721 -- top offender was Crustle, 4 losses) found
    Munkidori's Adrena-Brain firing ZERO times in 3 of those 4 games, despite the module
    docstring's own mined evidence (item 3.6) that an expert reference fires it ~11x/game.
    Mechanism: Adrena-Brain requires "any {D} energy" attached to Munkidori just to turn
    the Ability ON at all -- but _energy_guard (below, item 4 in the chain) greedily grabs
    ANY {D}-attach option the instant our active Grimmsnarl ex is under 2 energy, which is
    true for most of the early-mid game. If Munkidori and an under-energized Grimmsnarl are
    both in play, the single per-turn manual energy attach always goes to Grimmsnarl, and a
    0-energy Munkidori can never activate Adrena-Brain -- exactly the "ranker/guard
    mis-prioritizes an under-represented option class" failure pattern the other guards in
    this file were written to fix (see module docstring items 3/3.5/3.6), just one guard
    away from being caught: _adrena_trigger_guard already tries to FIRE the ability every
    turn it's available, but was never wired to the ENERGY that switches it on in the first
    place.

    Gate: [майнинг v062] item 1 -- REMOVED (was `_detect_matchup_raw(obs_dict) in {crustle,
    archaludon, lucario, tr_mewtwo, abomasnow}`, excluding alakazam/mirror/generic).
    docs/gold/v062_expert_playbook_lines.md lines 14-19: the expert reference sends the
    game's FIRST energy to Munkidori in EVERY matchup bucket (48-80% of ordinal-1 attaches,
    median turn 1-2 everywhere), not just grind matchups. Lines 55-57: the single starkest
    won/lost Adrena-Brain-volume gap in the whole mined dataset is vs Alakazam specifically
    (mean 4.21 activations/game won vs 2.15 lost, ~2x) -- exactly the matchup the old gate
    excluded. This guard now fires in every matchup; `_detect_matchup_raw` is still called
    at the call site purely for the [dbg] log tag (see agent()'s step-4 call), not to gate.

    Self-limit: only matches a Munkidori with 0 {D} energy attached (checked live off the
    obs_dict, per-location) -- the instant it has >=1, this guard stops matching that
    Munkidori (Adrena-Brain only needs ONE {D} to switch on; a 2nd is just parking, same
    judgment call as attach_target_score in scoring_grimmsnarl.py). Reuses _energy_guard's
    own option-shape convention (OPT_ATTACH, area==AREA_HAND, hand card id==Basic {D}
    Energy, inPlayArea identifies the destination) rather than inventing a new one.

    Returns None (falls through to _energy_guard / the scorer / trivial fallback --
    never-throw preserved) if no 0-energy Munkidori is in play, or no qualifying ATTACH
    option targets one.
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

        # Locate every 0-energy Munkidori currently in play (active + bench).
        zero_energy_active = False
        zero_energy_bench_idx = set()
        my_active = (my_pl.get("active") or [None])[0]
        if isinstance(my_active, dict) and int(my_active.get("id") or 0) == MUNKIDORI_ID:
            if len(my_active.get("energies") or []) == 0:
                zero_energy_active = True
        for idx, pk in enumerate(my_pl.get("bench") or []):
            if isinstance(pk, dict) and int(pk.get("id") or 0) == MUNKIDORI_ID:
                if len(pk.get("energies") or []) == 0:
                    zero_energy_bench_idx.add(idx)
        if not zero_energy_active and not zero_energy_bench_idx:
            return None

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
            if in_area == AREA_ACTIVE and zero_energy_active:
                return [i]
            if in_area == AREA_BENCH:
                in_idx = int(o.get("inPlayIndex") if o.get("inPlayIndex") is not None else -1)
                if in_idx in zero_energy_bench_idx:
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

    [майнинг v062] item 2: also sets `_adrena_intent` to one of "kill" (branch a), "combo"
    (branch b), or "value" (branch c) -- consumed by _adrena_sequence_guard's "await_target"
    step to decide whether the final snipe target stays the active (kill/combo, unchanged)
    or moves to the opponent's bench (value, new -- see that guard's own docstring and
    docs/gold/v062_expert_playbook_lines.md lines 58-61/124-149).
    """
    global _adrena_state, _adrena_intent, _n_adrena_tank_opportunistic, _n_adrena_fml_adjusted
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
            _adrena_intent = "kill"  # [майнинг v062] item 2 -- direct kill, target stays active
            return [ability_idx]

        if my_active and int(my_active.get("id") or 0) == GRIMMSNARL_ID:
            dark_energy = sum(
                1 for e in (my_active.get("energies") or []) if int(e) == BASIC_D_ENERGY_ID
            )
            if dark_energy >= 2:
                shadow_bullet_dmg = (_DAMAGE_TABLE.get(GRIMMSNARL_ID) or [180])[0] if _damage_table_ok else 180
                if opp_active_id == CRUSTLE_ID:
                    shadow_bullet_dmg = 0  # item 3.6(a): Rock Inn blocks Shadow Bullet vs active Crustle.
                # [ТЗ v074 P0] Under Full Metal Lab the same attack lands 150, so the
                # "snipe now, KO with Shadow Bullet next" window is 30 narrower. Firing the
                # once-per-turn ability on a combo that no longer closes wastes it. (The
                # direct-kill branch above is untouched: Adrena-Brain MOVES damage counters,
                # it is not damage from an attack, so the stadium does not apply to it.)
                _fml_cut = _stadium_damage_reduction(cur, opp_active_id)
                if shadow_bullet_dmg > 0 and _fml_cut > 0:
                    shadow_bullet_dmg = max(0, shadow_bullet_dmg - _fml_cut)
                    _n_adrena_fml_adjusted += 1
                    _dbg(f"adrena: Full Metal Lab cuts Shadow Bullet to {shadow_bullet_dmg} "
                         f"vs {{M}} active id={opp_active_id} (opp_hp={opp_hp}, "
                         f"snipe={available_snipe})")
                if shadow_bullet_dmg > 0 and opp_hp > shadow_bullet_dmg and opp_hp <= shadow_bullet_dmg + available_snipe:
                    _adrena_state = "await_from"
                    _adrena_intent = "combo"  # [майнинг v062] item 2 -- KO combo, target stays active
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
        _adrena_intent = "value"  # [майнинг v062] item 2 -- opportunistic, target routes to bench
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
    Resets `_adrena_state` (and `_adrena_intent`) to None on completion,
    desync, or any failure (never-throw preserved; a reset just lets the
    normal chain run for this decision instead of guessing).

    [майнинг v062] item 2 -- the final "await_target" step (ctx CTX_DAMAGE_COUNTER, where
    the counters actually land) now branches on `_adrena_intent`:
      - "kill" / "combo" (set by _adrena_trigger_guard branches a/b): UNCHANGED from wave 1
        -- target stays the opponent's ACTIVE, since that target was already fixed when the
        guard decided to fire (a guaranteed KO or KO-combo this turn).
      - "value" (branch c, opportunistic): target routes to the opponent's BENCH by default.
        docs/gold/v062_expert_playbook_lines.md lines 58-61 (vs Alakazam: Adrena-Brain
        targets are opponent-bench 65.4%/79.5% of the time in won/lost games) and lines
        124-149 (Archaludon reconstruction: damage routed onto not-yet-evolved backup
        Duraludon copies on the bench -- "kill the reinforcement before it evolves") both
        point the same direction. Priority among bench candidates: (1) a target this
        transfer KOs outright (remaining HP <= counters-moved*10, i.e. `_adrena_move_count`
        from the "await_count" step), preferring the LOWEST remaining HP among those
        (an evolving pre-evo/basic is typically the lowest-HP body on a bench anyway, so
        this already captures "prefer evolving basics" without a hardcoded card-id list);
        (2) else the single weakest (lowest remaining HP) bench mon; (3) fallback to active
        only if the engine offers no bench CARD options at all for the opponent this step.
    """
    global _adrena_state, _adrena_intent, _adrena_move_count, _n_adrena_steps
    if _adrena_state is None:
        return None
    try:
        sel = obs_dict.get("select") or {}
        ctx = int(sel.get("context") if sel.get("context") is not None else -1)
        opts = sel.get("option") or []
        if not opts:
            _adrena_state = None
            _adrena_intent = None
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]
        opp_pl = players[1 - me]

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
            if best_i is None:
                _adrena_intent = None
                return None
            _n_adrena_steps += 1
            return [best_i]

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
            if best_i is None:
                _adrena_intent = None
                return None
            # [майнинг v062] item 2: remember how many counters we're moving (0-3) -- the
            # "await_target" step needs this*10 to know how much HP a bench candidate dies to.
            _adrena_move_count = max(0, best_n)
            _n_adrena_steps += 1
            return [best_i]

        if _adrena_state == "await_target" and ctx == CTX_DAMAGE_COUNTER:
            opp_idx = 1 - me
            best_i = None
            if _adrena_intent in ("kill", "combo"):
                # [майнинг v062] item 2: unchanged from wave 1 -- these branches already
                # committed to a guaranteed KO/combo on the active this turn.
                for i, o in enumerate(opts):
                    if int(o.get("type") or -1) != OPT_CARD:
                        continue
                    if int(o.get("playerIndex") if o.get("playerIndex") is not None else -1) != opp_idx:
                        continue
                    area = int(o.get("area") if o.get("area") is not None else -1)
                    if area == AREA_ACTIVE:
                        best_i = i
                        break
            else:
                # [майнинг v062] item 2: "value" (opportunistic) branch -- route to bench.
                moved_hp = _adrena_move_count * 10
                bench_candidates = []  # (remaining_hp, option_index)
                active_i = None
                for i, o in enumerate(opts):
                    if int(o.get("type") or -1) != OPT_CARD:
                        continue
                    if int(o.get("playerIndex") if o.get("playerIndex") is not None else -1) != opp_idx:
                        continue
                    area = int(o.get("area") if o.get("area") is not None else -1)
                    if area == AREA_ACTIVE:
                        if active_i is None:
                            active_i = i
                        continue
                    if area != AREA_BENCH:
                        continue
                    idx = int(o.get("index") if o.get("index") is not None else -1)
                    pk = _resolve_pokemon(opp_pl, area, idx)
                    if not isinstance(pk, dict):
                        continue
                    bench_candidates.append((int(pk.get("hp") or 0), i))
                if bench_candidates:
                    killable = [c for c in bench_candidates if c[0] <= moved_hp]
                    pool = killable if killable else bench_candidates
                    pool.sort(key=lambda c: c[0])
                    best_i = pool[0][1]
                    _dbg(f"adrena target (value): bench i={best_i} hp={pool[0][0]} "
                         f"moved_hp={moved_hp} killable={bool(killable)} n_candidates={len(bench_candidates)}")
                elif active_i is not None:
                    best_i = active_i
                    _dbg("adrena target (value): no bench CARD options offered, fell back to active")
            if best_i is None:
                for i, o in enumerate(opts):
                    if int(o.get("type") or -1) == OPT_CARD and int(
                        o.get("playerIndex") if o.get("playerIndex") is not None else -1
                    ) == opp_idx:
                        best_i = i
                        break
            _adrena_state = None
            _adrena_intent = None
            if best_i is not None:
                _n_adrena_steps += 1
                return [best_i]
            return None

        # Any other context while a snipe is pending is an unexpected desync
        # (e.g. an unforeseen intervening decision) -- reset and let the
        # normal guard chain / ranker handle this decision safely.
        _adrena_state = None
        _adrena_intent = None
        return None
    except Exception:
        _adrena_state = None
        _adrena_intent = None
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


def _select_signature(obs_dict: dict) -> str | None:
    """[форензик v063 ДЕФЕКТ-6] md5 of the full (select, current) pair. If NOTHING in the
    select options or the board state changed between two consecutive calls, answering the
    same way again cannot make progress -- that is the loop the breaker detects. Any real
    game progress (energy attached, damage, hand/deck/prize counts, turnActionCount)
    changes `current`, so legitimate repeated-looking selects never match."""
    try:
        payload = json.dumps(
            [obs_dict.get("select"), obs_dict.get("current")],
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()
    except Exception:
        return None


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
      4. Munkidori energy guard ([форензик v062] item 3, NEW): in grind/tank
         matchups (crustle/archaludon/lucario/tr_mewtwo/abomasnow), force the
         turn's {D} energy attach onto a 0-energy Munkidori instead of letting
         step 5 (energy-attach guard) always claim it for Grimmsnarl ex --
         Adrena-Brain needs at least 1 {D} just to switch on, and the mined
         expert reference fires it ~11x/game (module docstring item 3.6) but
         v061 fired it 0 times in most live Crustle losses. See
         _munkidori_energy_guard's own docstring for the full diagnosis.
      5. Energy-attach guard (active Grimmsnarl ex needs {D}{D} -> attach it).
      6. Deterministic heuristic scorer (scoring_grimmsnarl.choose_options).
      7. Trivial fallback (first minCount indices).
    """
    global _n_model_decisions, _n_ko_guard, _n_promo_guard, _n_energy_guard
    global _n_adrena_trigger, _n_adrena_steps, _n_trivial_fallback, _exception_count
    global _n_adrena_tank_opportunistic, _n_ko_guard_crustle_blocked
    global _n_munkidori_energy_guard, _n_loop_breaker, _loop_sig, _loop_repeat

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        # New game: reset per-game guard/scorer state.
        global _adrena_state, _adrena_intent, _adrena_move_count, _raw_last_matchup
        _adrena_state = None
        _adrena_intent = None  # [майнинг v062] item 2
        _adrena_move_count = 0  # [майнинг v062] item 2
        _raw_last_matchup = None  # [форензик v062] item 2: reset the raw-dict sticky matchup too
        _loop_sig = None  # [форензик v063 ДЕФЕКТ-6]
        _loop_repeat = 0
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

    # [форензик v063 ДЕФЕКТ-6] Loop-breaker: the engine re-presenting a byte-identical
    # (select, current) pair means our previous (deterministic, hence identical) answer
    # made no progress. On the 3rd consecutive identical prompt start rotating the pick so
    # a different option is tried each round instead of stalling to the step limit.
    try:
        _sig = _select_signature(obs_dict)
        if _sig is not None and _sig == _loop_sig:
            _loop_repeat += 1
        else:
            _loop_sig = _sig
            _loop_repeat = 0
        if _loop_repeat >= 2:
            _n_loop_breaker += 1
            shift = (_loop_repeat - 1) % n_options
            k = min(max(1, min_count), max_count, n_options)
            pick = [(i + shift) % n_options for i in range(k)]
            _dbg(f"guard=loop_breaker repeat={_loop_repeat} shift={shift} pick={pick}")
            _dbg_alt("loop_breaker", sel.get("option") or [])
            return pick
    except Exception:
        pass

    # 0. Adrena-Brain sequence guard (forced continuation, no-op if not pending)
    try:
        seq = _adrena_sequence_guard(obs_dict)
        if seq is not None and all(0 <= i < n_options for i in seq):
            _dbg(f"guard=adrena_seq pick={seq} (ctx={int(sel.get('context') or -1)})")
            _dbg_alt("adrena_seq", sel.get("option") or [])
            return seq
    except Exception:
        pass

    # 1. Active-promotion guard
    try:
        promo = _promotion_guard(obs_dict)
        if promo is not None and all(0 <= i < n_options for i in promo):
            _n_promo_guard += 1
            _dbg(f"guard=promotion pick={promo} (ctx={int(sel.get('context') or -1)})")
            _dbg_alt("promotion", sel.get("option") or [])
            return promo
    except Exception:
        pass

    # 2. Lethal-KO guard
    try:
        guard = _ko_guard(obs_dict)
        if guard is not None and all(0 <= i < n_options for i in guard):
            _n_ko_guard += 1
            _dbg(f"guard=ko_lethal_gamewinning pick={guard}")
            _dbg_alt("ko_lethal_gamewinning", sel.get("option") or [])
            return guard
    except Exception:
        pass

    # 3. Adrena-Brain trigger guard (initiate the Munkidori snipe combo)
    try:
        adrena = _adrena_trigger_guard(obs_dict)
        if adrena is not None and all(0 <= i < n_options for i in adrena):
            _n_adrena_trigger += 1
            _dbg(f"guard=adrena_trigger pick={adrena}")
            _dbg_alt("adrena_trigger", sel.get("option") or [])
            return adrena
    except Exception:
        pass

    # 4. Munkidori energy guard ([форензик v062] item 3)
    try:
        munk = _munkidori_energy_guard(obs_dict)
        if munk is not None and all(0 <= i < n_options for i in munk):
            _n_munkidori_energy_guard += 1
            _dbg(f"guard=munkidori_energy pick={munk} matchup={_detect_matchup_raw(obs_dict)}")
            _dbg_alt("munkidori_energy", sel.get("option") or [])
            return munk
    except Exception:
        pass

    # 5. Energy-attach guard
    try:
        energy = _energy_guard(obs_dict)
        if energy is not None and all(0 <= i < n_options for i in energy):
            _n_energy_guard += 1
            _dbg(f"guard=energy_attach pick={energy}")
            _dbg_alt("energy_attach", sel.get("option") or [])
            return energy
    except Exception:
        pass

    # 6. Deterministic heuristic scorer
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

    # 7. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
