"""ogerpon_teal_tables.py — per-deck deck_tables for the Teal Mask Ogerpon ex "committed
one-shot plan" pilot (src/pilot/committed_plan.py). Phase 2.3.

Deck (decks/candidates/ogerpon_teal.csv, 60 cards): Teal Mask Ogerpon ex(96)x4 is the
ONLY Pokemon in the entire list — no evolution card of any kind, this deck is single-
attacker start to finish. Its own ability "Teal Dance" self-attaches a Basic {G} Energy
once per turn (and draws a card when it does), and its only attack "Myriad Leaf Shower"
costs {G}{G}{G} for a base-30 hit that scales with energy in play (see damage-correction
guard below). Energy package: Basic {G} Energy(1)x17 + Grow Grass Energy(18, +20 HP)x2 =
19 energy cards, backed by Bug Catching Set(1094)x4, Energy Retrieval(1118)x2, Energy
Search(1119)x4 — unusually energy-dense/consistent for a mono-color deck. Support:
Pokegear 3.0(1122)x2, Tera Orb(1127)x2 (tutors a Tera Pokemon — i.e. more copies of
Ogerpon itself), Tool Scrapper(1137)x1, Jumbo Ice Cream(1147)x2, Hero's Cape(1159)x1
(+100 HP tool), Boss's Orders(1182)x3, Briar(1201)x1, Judge(1213)x4, N's Plan(1221)x1,
Harlequin(1223)x2, Lillie's Determination(1227)x2, Lively Stadium(1251)x2 (+30 HP to all
Basics, relevant since Ogerpon itself is Basic).

## consider_bench_snipe

True. This deck's only attack, Myriad Leaf Shower, never hits the opponent's bench on its
own (no such text). The reason to turn this on is the SAME structural one as
lopunny_tables.py: 3x Boss's Orders(1182), and committed_plan.py's serves_plan() only
scores a Boss's Orders PLAY as "serving the plan" when plan.target_area is the bench
(requires this flag). It matters MORE here than for Lopunny: this deck has no fallback
attacker at all, so whenever the opponent's Active fully walls Myriad Leaf Shower (see
guard 2 below — a Rock-Inn Crustle makes it a guaranteed 0), redirecting to ANY other
target via Boss's Orders is the only way this deck does damage that turn at all.

## Named guards (matchup_guards)

Same option-(a) design as lopunny_tables.py: committed_plan.py's deck_tables now accepts
"matchup_guards": list[Callable[[GuardContext], tuple[int, float]]], applied per-candidate
inside _candidates_for_attacker (see that module's docstring / GuardContext dataclass).
See lopunny_tables.py's docstring for why option (a) was chosen over a bolt-on wrapper
function (option b): the guard needs the SAME already-computed, per-candidate context
(attacker/attack/target + Rock-Inn/weakness-adjusted base damage) that only exists inside
that loop, and it needs to be able to change which candidate wins the argmax, not just
tweak the plan after the fact.

There is no live way to classify the opponent's full archetype mid-game (see
lopunny_tables.py's docstring for the same point) — both guards below key off specific
card IDs actually visible on the opponent's board.

1. Myriad Leaf Shower energy-scaling damage correction (self, not opponent-triggered).
   The card text: "This attack does 30 more damage for each Energy attached to both
   Active Pokemon." Attack.damage from the engine is always just the printed base number
   (30) — see committed_plan.py's Engine notes, conditional bonus text is never applied
   automatically. This is the deck's ONLY attacker, so leaving this uncorrected means
   build_plan() undervalues its single win condition on almost every turn (Teal Dance
   self-attaches {G} energy every turn, so the attacker alone routinely carries several
   energy by the time it swings). Correction: +30 damage for every Energy card attached
   to the attacker PLUS every Energy card attached to the target, i.e.
   30 * (len(attacker.energies) + len(target.energies)) on top of the listed 30 — applied
   only when target_area is the opponent's Active (the card text says "both ACTIVE
   Pokemon"; a Boss's-Orders bench-snipe target is not covered by this clause, so the
   correction is skipped there and the plan sees only the listed 30 against a benched
   target, which is correct-but-conservative).

2. Crustle-wall redirect guard (opponent's target is an active, Rock-Inn Crustle — id
   345, committed_plan.ROCK_INN_CRUSTLE_ID, reused directly rather than re-hardcoded here
   to avoid the two constants drifting apart). Unlike Mega Lopunny ex (see
   lopunny_tables.py — CardData.ex == False for a Mega ex, so Rock Inn's ex-check misses
   it entirely), Teal Mask Ogerpon ex genuinely has CardData.ex == True (verified live
   against the engine, see the smoke test), so committed_plan.py's own generic Rock-Inn
   handling already, correctly, zeroes Myriad Leaf Shower's damage against an active
   Crustle — nothing to fix in the damage math itself. But because this deck has no other
   attacker, build_plan's argmax would otherwise still happily "commit" to attacking a
   wall for 0 real damage (0 still out-scores a non-lethal bench_target_penalty(-400) hit
   elsewhere), wasting the turn. This matches FIELD_RECON_2026-08-02.md's matchup matrix
   (б): ogerpon_teal vs crustle = 0.0% WR, 0/26 games (N=26, marked * per the report's
   N<150 caveat, but 0-for-26 needs no statistical rescue given the mechanical certainty
   above). Guard: when the candidate's target is that specific active Crustle AND its
   base_damage is already 0 (checked directly rather than re-deriving "is this actually
   blocked" from the ex-flag, so the guard stays correct even if the core's Rock-Inn logic
   ever changes), apply a score penalty large enough to lose to almost any real-damage
   alternative (e.g. a Boss's-Orders bench snipe) but far below the lethal bonus, so a
   genuinely lethal line through Crustle itself (extremely unlikely given the wall, but
   not impossible if some future correction restores nonzero damage) is never blocked by
   this guard.

Other cells in the same matchup row (mirror_grimmsnarl 83.6%/N=232 — the one TRUSTWORTHY,
N>=150 cell in this deck's whole row, strongly favorable; tr_mewtwo 80.0%/N=10*,
garchomp 88.2%/N=17* also favorable; alakazam 21.1%/N=19*, dipplin 33.3%/N=12*,
dragapult 16.7%/N=6*, mega_kangaskhan 45.5%/N=11*, lopunny 7.0%/N=43* unfavorable) were
NOT turned into guards: the favorable ones have no actionable behavior change to encode
(a good matchup just means "play normally"), and the unfavorable ones besides crustle
have no accompanying card-text mechanism explaining WHY (unlike Crustle's literal damage
wall) at N too small (6-19 games) to trust as a standalone signal per the report's own
caveat — encoding them would be fitting noise, not a matchup fact.
"""
from __future__ import annotations

from pathlib import Path

try:
    from cg_native.api import AreaType  # type: ignore
except Exception:
    from cg.api import AreaType  # type: ignore

from pilot.committed_plan import GuardContext, ROCK_INN_CRUSTLE_ID

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DECK_CSV = _REPO_ROOT / "decks" / "candidates" / "ogerpon_teal.csv"

MYRIAD_LEAF_SHOWER_ATTACK_ID = 120   # Teal Mask Ogerpon ex (96) — base 30, +30/energy on both actives
CRUSTLE_WALL_PENALTY = 2000.0


def _load_deck() -> list[int]:
    return [int(line.strip()) for line in _DECK_CSV.read_text().splitlines() if line.strip()]


def _myriad_leaf_shower_energy_bonus(ctx: GuardContext) -> tuple[int, float]:
    if ctx.attack_id != MYRIAD_LEAF_SHOWER_ATTACK_ID:
        return 0, 0.0
    if ctx.target_area != int(AreaType.ACTIVE):
        return 0, 0.0  # card text is "both ACTIVE Pokemon" — doesn't apply to a bench snipe
    if ctx.base_damage == 0:
        # BUGFIX (see _crustle_wall_guard below): base_damage is already Rock-Inn/weakness/
        # resistance-adjusted by committed_plan._leaf_damage BEFORE any guard runs. Myriad
        # Leaf Shower's LISTED base is always 30 (never 0 on its own), so base_damage==0
        # here means the Rock-Inn-Crustle wall (committed_plan.ROCK_INN_CRUSTLE_ID) already
        # blocked this attack entirely. Without this check, this guard used to add
        # 30*bonus_energy back on TOP of that zeroed damage regardless of the wall — turning
        # a genuinely-blocked (0 real damage) attack into a candidate the build_plan argmax
        # saw as high-damage or even lethal (is_lethal computed from the POST-guard damage,
        # after both guards' damage_delta are summed) and committed to every turn, while the
        # real engine actually dealt 0 damage every time (verified live: 3/3 games, every
        # single Myriad Leaf Shower executed against an active Rock-Inn Crustle(345) did
        # real_dmg=0 despite committed_plan logging dmg=90..270 lethal=True). That inflated
        # score also swamped _crustle_wall_guard's -2000 penalty via the +100,000 lethal
        # bonus, defeating the wall guard's entire purpose. Skipping the bonus here restores
        # base_damage==0 as the guard's actual final damage, so _crustle_wall_guard's penalty
        # can do its job.
        return 0, 0.0
    bonus_energy = len(ctx.attacker.energies or []) + len(ctx.target.energies or [])
    return 30 * bonus_energy, 0.0


def _crustle_wall_guard(ctx: GuardContext) -> tuple[int, float]:
    if ctx.target_cdata is not None and ctx.target_cdata.cardId == ROCK_INN_CRUSTLE_ID and ctx.base_damage == 0:
        return 0, -CRUSTLE_WALL_PENALTY
    return 0, 0.0


def get_deck_tables() -> dict:
    return {
        "deck": _load_deck(),
        "consider_bench_snipe": True,
        "matchup_guards": [_myriad_leaf_shower_energy_bonus, _crustle_wall_guard],
        # Below default (500): mono-{G}, unusually energy-dense (19/60 energy cards) and
        # backed by 3 separate energy-fetch effects (Bug Catching Set/Energy Retrieval/
        # Energy Search) — a needs_energy plan here resolves far more reliably than the
        # generic default assumes.
        "energy_step_penalty": 400.0,
        # Default (300), effectively a dead code path: the deck's sole Pokemon (96) is
        # Basic with no evolution card anywhere in the list, so needs_evolution can never
        # become True for this deck's own build_plan candidates — left at default rather
        # than tuned, since there is nothing here to tune.
        "evolution_penalty": 300.0,
        # Default (400): no deck-specific reason found to deviate.
        "bench_target_penalty": 400.0,
        # Default (20): Ogerpon Teal ex is Grass-type; no field data pointing at a
        # Fire-resistance interaction worth deviating on.
        "resistance_reduction": 20.0,
    }


DECK_TABLES = get_deck_tables()
