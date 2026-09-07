"""lopunny_tables.py — per-deck deck_tables for the Mega Lopunny ex "committed one-shot
plan" pilot (src/pilot/committed_plan.py). Phase 2.3.

Deck (decks/candidates/lopunny.csv, 60 cards): Buneary(848)x4 -> Mega Lopunny ex(849)x4
(single 1-stage evolution line, the deck's only real attacker), Dunsparce(305)x4 ->
Dudunsparce(66)x4 (secondary attacker/draw engine), Fan Rotom(174)x1 (early Basic search),
Mist/Enriching/Spiky Energy(11/13/14)x4 each (12 Special Energy cards providing {C}, NO
Basic Energy at all), Buddy-Buddy Poffin(1086)x4 + Ultra Ball(1121)x4 + Pokegear
3.0(1122)x4 + Poke Pad(1152)x4 (Pokemon search/setup), Air Balloon(1174)x4 (retreat cost
reduction), Boss's Orders(1182)x3, Xerosic's Machinations(1197)x1, Hilda(1225)x4 (tutors
an Evolution Pokemon AND an Energy card together), Lillie's Determination(1227)x4 (hand
refresh/draw), Wally's Compassion(1229)x4 (heals + returns energy from a Mega ex).

## consider_bench_snipe

True. NOT because any of Buneary/Dunsparce/Dudunsparce's own attacks hit the opponent's
bench directly — verified against data/EN_Card_Data.csv they do not: "Run Around"(848)/
"Trading Places"(305) both just SWITCH the user with one of ITS OWN bench slots, and
Dudunsparce's "Land Crush"/Buneary's "Kick" only ever target the opponent's Active, same
as everything else in this deck. (The task brief's own example naming these three as
snipe attackers does not hold up against the actual card text — flagging the correction
here rather than silently deviating from the brief.)
The real reason: this deck runs 3x Boss's Orders (1182), and committed_plan.py's
serves_plan() only ever scores a Boss's Orders PLAY as "serving the plan" when
plan.target_area is the opponent's BENCH (see serves_plan's PLAY branch) — which itself
only happens when build_plan() was allowed to consider bench targets at all, i.e. when
this flag is on. Spiky Hopper alone (160 flat, unaffected by the target's own effects)
is enough to one-shot most low/mid-HP bench setup pieces in the current field (Dreepy 70,
Impidimp 70, Kadabra 80, Gible 70, Abra 50, ...), so "Boss's Orders + attack a benched
threat instead of the wall in front of us" is a real, sometimes-lethal line this deck
should be allowed to consider, not a purely per-deck non-decision that the deck-agnostic
core should stay silent on. build_plan's own bench_target_penalty + the huge lethal bonus
already keep non-lethal snipes from being over-picked, so turning this on is low-risk.

## Named guards (matchup_guards)

Design choice — option (a) from the task brief: committed_plan.py got ONE minimal,
additive extension (deck_tables["matchup_guards"]: list[Callable[[GuardContext],
tuple[int, float]]], applied per-candidate inside _candidates_for_attacker, see that
module's docstring and the GuardContext dataclass for the exact contract). Rejected
option (b) (a bolt-on function the submission wrapper calls before/after choose_options)
because a matchup guard's natural inputs — this SPECIFIC candidate's attacker/attack/
target and its already-weakness/Rock-Inn-adjusted damage — only exist inside
_candidates_for_attacker's loop; reconstructing them from outside would mean either
duplicating _leaf_damage's math in this file (drifts out of sync with the core) or only
being able to nudge the ALREADY-CHOSEN plan post-hoc (can't change *which* attack/target
wins the argmax, which is exactly what both guards below need to do). The core extension
is ~20 lines, additive-only (deck_tables["matchup_guards"] defaults to an empty list, so
every other deck_tables user is unaffected), and mirrors the existing deck_tables
plumbing pattern already in the module.

There is no live way to classify the opponent's full archetype from a partial, in-progress
board (scripts/classify_opponent_archetype.py's classify() needs the full 60-card decklist
from a finished replay — not available mid-game). So both guards below key off SPECIFIC
card IDs actually visible on the opponent's board, per the task brief's own fallback
suggestion, not an archetype label.

1. Gale Thrust bench-move damage correction (self, not opponent-triggered). Mega Lopunny
   ex's "Gale Thrust" (attackId 1225) reads: "If this Pokemon moved from your Bench to the
   Active Spot this turn, this attack does 170 more damage" (60 -> 230). Per
   committed_plan.py's own docstring, Attack.damage from the engine is always just the
   printed BASE number (60 here) — conditional bonus text is never applied automatically.
   Whenever plan.attacker_area is BENCH, committed_plan's serves_plan() already forces a
   RETREAT to bring that bench attacker into the Active Spot before the ATTACK step (see
   serves_plan's RETREAT branch: "plan.attacker_area == BENCH -> make room to bring the
   planned bench attacker up"), so "attacker started this turn on the bench" is reliably
   true by the time Gale Thrust fires under a committed plan built with attacker_area ==
   BENCH. Without this correction, build_plan can never see the 230-damage line at all and
   will always rank the flat-160 Spiky Hopper above a lethal-only-with-the-bonus Gale
   Thrust, missing real kills.

2. Alakazam-line guard (opponent board has any of Abra(741)/Kadabra(742)/Alakazam(743)).
   Alakazam's own attack "Powerful Hand" ({P}, listed damage 0) actually does
   "Place 2 damage counters on your opponent's Active Pokemon for each card in your
   hand" — i.e. its real damage is a function of the OPPONENT's hand size, completely
   invisible to this deck's own damage model, and both of its pre-evolutions carry a
   "Psychic Draw" ability that refills that hand every turn it evolves. This matches
   FIELD_RECON_2026-08-02.md's matchup matrix (б): lopunny vs alakazam = 29.4% WR (N=34,
   marked * / not decisive alone per the report's own N<150 caveat, but directionally
   consistent with a mechanic this deck's damage math structurally cannot see coming).
   Guard: when an Alakazam-line card is visible anywhere on the opponent's board, apply a
   score penalty to any candidate that ISN'T ready to fire THIS turn (needs_energy or
   needs_evolution truthy) — every extra turn we spend "setting up" is a turn that funds
   Powerful Hand's hand-size scaling for free, so the plan should prefer the best attack
   available NOW over a bigger number next turn more strongly than the generic defaults
   already do.

Other cells in the same matchup row (crustle 80.9%/N=47*, tr_mewtwo 70.6%/N=17*,
dipplin 90.0%/N=20*, garchomp 33.3%/N=21*, dragapult 41.2%/N=17*,
mega_kangaskhan 40.0%/N=10*) were deliberately NOT turned into guards:
  - crustle: already fully handled by the GENERIC core, no guard needed. Mega Lopunny ex
    has CardData.ex == False (only megaEx == True — verified live against the engine, see
    the smoke test / submissions/v051_lopunny_crustle/main.py:108-124's prior verification
    note), so Crustle's Mysterious Rock Inn ability-check (`attacker_cdata.ex`) never
    triggers against EITHER of Lopunny's attacks in the first place — Gale Thrust already
    deals full listed damage against Crustle even without Spiky Hopper's explicit pierce
    text. Nothing to add here.
  - garchomp/dragapult/mega_kangaskhan/tr_mewtwo/dipplin: all N<150 (10-21 games) with no
    accompanying card-text mechanism found that would explain the direction of the number
    (unlike Alakazam's hand-size burst) — encoding a guard here would be fitting noise, not
    a matchup fact. Left alone per the task's own "not a decisive signal alone" caveat.
"""
from __future__ import annotations

from pathlib import Path

try:
    from cg_native.api import AreaType  # type: ignore
except Exception:
    from cg.api import AreaType  # type: ignore

from pilot.committed_plan import GuardContext

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DECK_CSV = _REPO_ROOT / "decks" / "candidates" / "lopunny.csv"

GALE_THRUST_ATTACK_ID = 1225      # Mega Lopunny ex (849) — base 60, +170 if moved from Bench
ALAKAZAM_LINE_IDS = {741, 742, 743}  # Abra / Kadabra / Alakazam (MEG)
GALE_THRUST_BENCH_BONUS = 170
ALAKAZAM_NOT_READY_PENALTY = 1500.0


def _load_deck() -> list[int]:
    return [int(line.strip()) for line in _DECK_CSV.read_text().splitlines() if line.strip()]


def _gale_thrust_bench_bonus(ctx: GuardContext) -> tuple[int, float]:
    if ctx.attack_id == GALE_THRUST_ATTACK_ID and ctx.attacker_area == int(AreaType.BENCH):
        return GALE_THRUST_BENCH_BONUS, 0.0
    return 0, 0.0


def _opp_has_any(obs, ids: set[int]) -> bool:
    """True if any of `ids` is visible anywhere on the opponent's current board (active
    or bench) — the only live, reliable substitute for full-decklist archetype
    classification (see module docstring)."""
    opp = obs.current.players[1 - obs.current.yourIndex]
    mons = list(opp.active or []) + list(opp.bench or [])
    return any(m is not None and m.id in ids for m in mons)


def _alakazam_urgency_guard(ctx: GuardContext) -> tuple[int, float]:
    if not (ctx.needs_energy or ctx.needs_evolution):
        return 0, 0.0
    if _opp_has_any(ctx.obs, ALAKAZAM_LINE_IDS):
        return 0, -ALAKAZAM_NOT_READY_PENALTY
    return 0, 0.0


def get_deck_tables() -> dict:
    return {
        "deck": _load_deck(),
        "consider_bench_snipe": True,
        "matchup_guards": [_gale_thrust_bench_bonus, _alakazam_urgency_guard],
        # Below default (500): only 12/60 cards are energy (Mist/Enriching/Spiky Special
        # Energy, no Basic Energy, no retrieval effect in this list) — low density argues
        # AGAINST relaxing this, so it's left at the generic default rather than lowered.
        "energy_step_penalty": 500.0,
        # Below default (300): single 1-stage line (Buneary -> Mega Lopunny ex), no
        # Stage 2 anywhere, and Hilda(1225, x4) tutors an Evolution Pokemon + an Energy
        # card in the SAME search, on top of Ultra Ball/Poke Pad. A needs_evolution plan
        # in this deck is materially cheaper to complete than the generic default assumes.
        "evolution_penalty": 200.0,
        # Default (400): no deck-specific reason found to deviate.
        "bench_target_penalty": 400.0,
        # Default (20): Lopunny's own attacks are Colorless-type, so this constant never
        # applies to our offense either way — irrelevant, left at default.
        "resistance_reduction": 20.0,
    }


DECK_TABLES = get_deck_tables()
