"""committed_plan.py — deck-agnostic "committed one-shot plan" pilot core.

Idea: once per our turn (on a MAIN select), pick ONE (attacker, attack, target) triple that
maximizes damage/lethality, then COMMIT to it — every subsequent select this turn (ATTACH,
EVOLVE, RETREAT, the Boss's Orders PLAY + its follow-up target pick, ...) is scored by how
much it *serves* that committed plan, instead of re-deciding from scratch each step. This
avoids the classic failure mode of a per-step greedy agent flip-flopping between candidate
lines mid-turn.

This module is deck-agnostic: it knows nothing about any specific deck's card IDs. It only
hard-codes GENERIC field mechanics that any deck can run into (Rock Inn / Full Metal Lab /
Spiky-Hopper-style "pierce" attacks / Boss's Orders), exactly like
submissions/v076_final/scoring_grimmsnarl.py does for its own deck-specific mechanics.

Per-deck tuning (weights, thresholds, matchup guards) is intentionally NOT here — that is
the next layer (src/pilot/decks/{deck}_tables.py). This module accepts an optional
``deck_tables: dict | None`` on every public entry point (``build_plan``, ``serves_plan``,
``choose_options``). If a caller does not want to pass it on every call, it can instead call
``set_deck_tables(tables)`` once and leave ``deck_tables=None`` on every subsequent call —
the module falls back to the last value passed to ``set_deck_tables``. Recognized keys (all
optional, all have sane defaults if absent):
  - "deck": list[int]              — REQUIRED to answer the deck-select step (60 card ids).
                                      This module has no built-in deck of its own.
  - "consider_bench_snipe": bool   — also enumerate the opponent's bench as plan targets
                                      (reachable only via a Boss's-Orders-style gust play).
                                      Off by default: a non-lethal gust trades away the whole
                                      turn's tempo for a maybe, which is a per-deck judgment
                                      call this generic core should not make on its own.
  - "energy_step_penalty", "evolution_penalty", "bench_target_penalty",
    "resistance_reduction", "retreat_lethal_threat_bonus", "retreat_heavy_threat_bonus" —
    override the scoring constants below.
  - "matchup_guards": list[Callable[[GuardContext], tuple[int, float]]] — per-deck
    adjustments to a SINGLE candidate Plan (one attacker/attack/target combination)
    that this generic core has no way to know about on its own: either a correction to
    the engine's LISTED (base, printed) damage number for an attack whose real damage
    is conditional (e.g. "does N more damage if this Pokemon moved from your Bench this
    turn", "does N more damage for each Energy attached to both Active Pokemon" —
    Attack.damage from the engine is always just the printed base number, see the
    Engine notes below), or an opponent-archetype-triggered bonus/penalty (e.g. "this
    matchup is a guaranteed 0-damage wall, prefer a different candidate"), keyed off
    card IDs actually visible on the opponent's board right now (there is no reliable
    way to classify the opponent's full archetype live from partial board info — a
    per-deck table's own docstring, e.g. src/pilot/decks/lopunny_tables.py, is where
    that tradeoff and the specific guards built on it are documented).
    Each callable receives a GuardContext (one candidate's full context: obs, attacker/
    attack/target ids, the already-weakness/Rock-Inn/Full-Metal-Lab-adjusted base
    damage, needs_energy/needs_evolution, ...) and returns (damage_delta, score_delta):
    damage_delta is added to the candidate's damage BEFORE is_lethal/score are computed
    (so it can turn a plan lethal), score_delta is added to the final score afterward
    (for adjustments that shouldn't affect the lethal/lethal-adjacent damage math, only
    tie-breaking preference). Off by default (empty list): every deck-specific fact a
    guard could encode is, definitionally, not something this module should guess at.

Engine notes (see module docstring history / task brief for the full verified fact sheet):
  - The engine binary is closed. Attack.damage from all_attack() is LISTED (base, printed-
    on-card) damage only — no weakness/resistance/ability/stadium modifiers. Those are
    applied by hand here, mirroring submissions/v076_final/scoring_grimmsnarl.py.
  - Weakness in this game's format is a x2 multiplier (not classic +20).
  - Resistance's magnitude is NOT exposed by the API (CardData.resistance is a TYPE only,
    no amount) and is nearly absent from the Pocket-format card pool. We apply the classic
    paper-TCG value (-20) when a resistance type is present and matches the attacker, but
    this magnitude is UNVERIFIED against this engine — flagged here and overridable via
    deck_tables["resistance_reduction"].
  - Status conditions (poisoned/burned/asleep/paralyzed/confused) are exposed as booleans on
    the active Pokemon but their effect on *which options the engine offers* is binary
    behavior we cannot introspect from Python. We do not try to emulate rules like "asleep/
    paralyzed can't retreat" — we only read the flags as context for a future per-deck layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# ── Engine import: try the cross-platform dev build (cg_native, picks the right .so/.dylib
# for the local OS/arch) first, then fall back to the plain `cg` package. Production
# submissions bundle a per-submission `cg/` folder that is ITSELF the cross-platform loader
# (see submissions/v076_final/cg/sim.py, identical to src/cg_native/sim.py), so `cg.api`
# resolves fine on Kaggle/Linux either way. On a local macOS dev box, src/cg/api.py's
# `.sim` unconditionally dlopen()s a Linux-only libcg.so and raises OSError at import time —
# cg_native.api is the one that actually loads here. Broad `except Exception` on purpose:
# the failure mode is OSError from ctypes, not ImportError.
try:
    from cg_native.api import (  # type: ignore
        AreaType, EnergyType, CardType, OptionType, SelectContext,
        Observation, State, PlayerState, Pokemon, Card, CardData, Attack, Option, SelectData,
        to_observation_class, all_card_data, all_attack,
    )
except Exception:
    from cg.api import (  # type: ignore
        AreaType, EnergyType, CardType, OptionType, SelectContext,
        Observation, State, PlayerState, Pokemon, Card, CardData, Attack, Option, SelectData,
        to_observation_class, all_card_data, all_attack,
    )

# ── scripts/gate_scout_guard.py's greedy_meta_agent — safety-net fallback, imported (not
# copied) per the task brief. Add scripts/ to sys.path the way the rest of the project does
# (e.g. scripts/build_deck_belief_data.py:64, scripts/crustle_tech_vs_v016.py:28).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from gate_scout_guard import greedy_meta_agent  # type: ignore
except Exception:
    def greedy_meta_agent(obs_dict: dict) -> list[int]:
        """Last-resort local stand-in if scripts/gate_scout_guard.py can't be imported
        (e.g. repo layout changed). Mirrors its trivial branch only: no attack-damage
        awareness, just "pick the first minCount options"."""
        sel = obs_dict.get("select") if obs_dict else None
        if sel is None:
            return []
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        min_c = int(sel.get("minCount") or 1)
        max_c = int(sel.get("maxCount") or 1)
        k = max(1, min(min_c, max_c, n))
        return list(range(k))


# ── Generic field mechanics (verified facts, NOT deck-specific — see module docstring) ───
ROCK_INN_CRUSTLE_ID = 345          # Crustle: Ability blocks ALL dmg from opponent's ex attacks
FULL_METAL_LAB_ID = 1244           # Stadium: {M} Pokemon take -30 dmg (after weakness), both sides
SPIKY_HOPPER_ATTACK_ID = 1226      # Mega Lopunny ex's "isn't affected by any effects..." attack
BOSS_ORDERS_CARD_ID = 1182         # Universal gust supporter, not deck-specific
_PIERCE_TEXT_MARKER = "affected by any effects on your opponent"  # normalized substring
_DEFAULT_RESISTANCE_REDUCTION = 20  # classic paper-TCG value; UNVERIFIED against this engine

_DEFAULT_ENERGY_STEP_PENALTY = 500.0
_DEFAULT_EVOLUTION_PENALTY = 300.0
_DEFAULT_BENCH_TARGET_PENALTY = 400.0
_LETHAL_BONUS = 100_000.0
# Defensive RETREAT bonuses (see _opponent_threat_damage / serves_plan's RETREAT branch).
# Scaled well below _LETHAL_BONUS (our own guaranteed KO is still the top priority: taking
# it doesn't need a "should I retreat instead" fight) but clearly above the existing
# plan-progress bonuses just above this (3000 bench-retreat / 5000 attach-fills-need /
# 5000 evolve / 4000 Boss's Orders) -- a real "we die next turn if we stay in" threat
# should outweigh "the plan isn't finished yet".
_DEFAULT_RETREAT_LETHAL_THREAT_BONUS = 9000.0
_DEFAULT_RETREAT_HEAVY_THREAT_BONUS = 2000.0
# When our committed ATTACK is NOT lethal this turn AND the opponent can already KO our
# active next turn (same one-hop _opponent_threat_damage check as the RETREAT branch above)
# AND a RETREAT option actually exists in this select, the normal 1,000,000 "execute planned
# attack" bonus is replaced by this much smaller one -- letting _DEFAULT_RETREAT_LETHAL_
# THREAT_BONUS (9000, plus RETREAT's own +10 base priority) win the argmax instead. Scaled
# well below the retreat-under-lethal-threat bonus (so retreating away from a guaranteed
# death actually wins) but still above ATTACK's own +50 base priority and the smaller
# plan-progress bonuses, so a non-lethal attack still beats truly unrelated options when
# retreat isn't the top scorer for some other reason. Does NOT apply when our attack IS
# lethal (finishing the opponent off always stays top priority) or when no RETREAT option
# exists this select (nothing to switch to -- deprioritizing the attack would just be
# harmful).
_DEFAULT_ATTACK_UNDER_LETHAL_THREAT_BONUS = 500.0

_OPTION_TYPE_PRIORITY = {
    int(OptionType.ATTACK): 50.0,
    int(OptionType.ABILITY): 40.0,
    int(OptionType.EVOLVE): 35.0,
    int(OptionType.PLAY): 30.0,
    int(OptionType.ATTACH): 20.0,
    int(OptionType.RETREAT): 10.0,
    int(OptionType.DISCARD): 8.0,
    int(OptionType.CARD): 5.0,
    int(OptionType.ENERGY): 5.0,
    int(OptionType.NUMBER): 5.0,
    int(OptionType.YES): 5.0,
    int(OptionType.END): 0.0,
}

_DEBUG_LOG = os.environ.get("CP_DEBUG", "1") != "0"


# ── CardData/Attack lookup tables (built once at import, like scoring_grimmsnarl.CARD_DB) ─
try:
    CARD_DB: dict[int, CardData] = {c.cardId: c for c in all_card_data()}
    ATTACK_DB: dict[int, Attack] = {a.attackId: a for a in all_attack()}
    _ENGINE_DB_OK = True
except Exception as _e:  # pragma: no cover - only if the engine binary itself is broken
    CARD_DB = {}
    ATTACK_DB = {}
    _ENGINE_DB_OK = False
    print(f"[cp] FATAL: could not load CARD_DB/ATTACK_DB at import: {_e!r}", file=sys.stderr)


# ── Diagnostics: never let an except silently return None/[] without counting+logging ─────
class _ExceptionStats:
    """Ported pattern from scripts/scan_daily_zip_field.py's SkipStats: count every
    exception by reason, print at most `max_tb` tracebacks per reason so a hot loop doesn't
    spam stderr into uselessness."""

    def __init__(self, max_tb: int = 3):
        self.counts: dict[str, int] = {}
        self._tb_shown: dict[str, int] = {}
        self.max_tb = max_tb

    def skip(self, reason: str, exc: BaseException | None = None) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        STATS["exceptions"] += 1
        shown = self._tb_shown.get(reason, 0)
        if exc is not None and shown < self.max_tb:
            print(f"[cp] exception reason={reason}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._tb_shown[reason] = shown + 1


_exc = _ExceptionStats()

# Module-level counters, readable externally for the exec-smoke test.
STATS: dict[str, object] = {
    "plan_built": 0,
    "plan_missing": 0,
    "generic_fallback": 0,
    "exceptions": 0,
    "served": {},  # OptionType name -> count of times a picked option was scored by serves_plan with bonus > 0
}


def _bump_served(type_name: str) -> None:
    served = STATS["served"]  # type: ignore[assignment]
    served[type_name] = served.get(type_name, 0) + 1


def _log(msg: str) -> None:
    if _DEBUG_LOG:
        try:
            print(f"[cp] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass


def _select_signature(obs_dict: dict) -> str | None:
    """Ported verbatim (pattern) from submissions/v076_final/main.py:1477-1491."""
    try:
        payload = json.dumps(
            [obs_dict.get("select"), obs_dict.get("current")],
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()
    except Exception:
        return None


# ── deck_tables plumbing ────────────────────────────────────────────────────────────────
_CURRENT_DECK_TABLES: dict | None = None


def set_deck_tables(tables: dict | None) -> None:
    """Set the module-level default deck_tables used whenever a caller passes
    deck_tables=None to build_plan/serves_plan/choose_options. A submission wrapper calls
    this once (e.g. at import time, with its own {deck}_tables.py module's dict)."""
    global _CURRENT_DECK_TABLES
    _CURRENT_DECK_TABLES = tables


def _tables(deck_tables: dict | None) -> dict:
    return deck_tables if deck_tables is not None else (_CURRENT_DECK_TABLES or {})


# ── Plan ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Plan:
    attacker_area: int
    attacker_slot: int
    attacker_id: int
    attack_id: int
    attack_index_hint: int
    target_area: int
    target_slot: int
    target_id: int
    expected_damage: int
    is_lethal: bool
    needs_energy: dict[int, int] | None   # EnergyType -> count still short. None = ready now.
    needs_evolution: bool
    score: float


@dataclass
class GuardContext:
    """Read-only context passed to every deck_tables["matchup_guards"] callable for ONE
    candidate (attacker, attack, target) triple, before it becomes a Plan. See the
    "matchup_guards" entry in the module docstring for the full contract."""
    obs: Observation
    attacker_area: int
    attacker_slot: int
    attacker_id: int
    attacker: Pokemon
    attack_id: int
    base_damage: int          # already weakness/Rock-Inn/Full-Metal-Lab-adjusted
    target_area: int
    target_slot: int
    target: Pokemon
    target_cdata: CardData | None
    needs_energy: dict[int, int] | None
    needs_evolution: bool


# ── Board accessors ─────────────────────────────────────────────────────────────────────
def _pokemon_at(ps: PlayerState, area: int, slot: int) -> Pokemon | None:
    if area == int(AreaType.ACTIVE):
        return ps.active[0] if ps.active else None
    if area == int(AreaType.BENCH):
        return ps.bench[slot] if ps.bench and 0 <= slot < len(ps.bench) else None
    return None


def _card_at(obs: Observation, area: int | None, index: int | None, player_index: int) -> Card | None:
    """Minimal port of submissions/v076_final/scoring_grimmsnarl.py:get_card — resolves a
    raw Card struct (id/serial only) at (area, index) for `player_index`."""
    if area is None or index is None:
        return None
    ps = obs.current.players[player_index]
    if area == int(AreaType.HAND):
        return ps.hand[index] if ps.hand and index < len(ps.hand) else None
    if area == int(AreaType.DECK):
        return obs.select.deck[index] if obs.select and obs.select.deck and index < len(obs.select.deck) else None
    if area == int(AreaType.DISCARD):
        return ps.discard[index] if ps.discard and index < len(ps.discard) else None
    if area == int(AreaType.ACTIVE):
        return ps.active[index] if ps.active and index < len(ps.active) else None
    if area == int(AreaType.BENCH):
        return ps.bench[index] if ps.bench and index < len(ps.bench) else None
    if area == int(AreaType.STADIUM):
        return obs.current.stadium[index] if obs.current.stadium and index < len(obs.current.stadium) else None
    return None


def _option_played_card(obs: Observation, opt: Option, player_index: int) -> Card | None:
    """Card an option refers to for PLAY/ATTACH/EVOLVE (the card being played/attached/
    evolved-into), read from (opt.area, opt.index) — same convention as option_card() in
    submissions/v076_final/scoring_grimmsnarl.py."""
    if opt.type == int(OptionType.PLAY):
        return _card_at(obs, int(AreaType.HAND), opt.index, player_index)
    return _card_at(obs, opt.area, opt.index, player_index)


# ── Damage math ──────────────────────────────────────────────────────────────────────────
def _normalize(text: str | None) -> str:
    return (text or "").replace("’", "'").lower()


def _is_pierce_attack(attack: Attack) -> bool:
    """Spiky-Hopper-style attacks that ignore ability/stadium modifiers on the opponent's
    Active (but NOT weakness/resistance — those are a different category of "effect" per
    the card text). attackId==1226 is a cheap fast-path; the text substring generalizes to
    any future deck with a similarly-worded attack."""
    if attack.attackId == SPIKY_HOPPER_ATTACK_ID:
        return True
    return _PIERCE_TEXT_MARKER in _normalize(attack.text)


def _leaf_damage(obs: Observation, attacker_id: int, attack: Attack, target: Pokemon,
                  target_cdata: CardData | None, deck_tables: dict) -> int:
    """Listed damage + weakness x2 + Rock Inn / Full Metal Lab / pierce-attack handling.
    Mirrors submissions/v076_final/scoring_grimmsnarl.py's effective_damage /
    stadium_damage_reduction / ex_damage_prevented, generalized to any deck."""
    base = attack.damage or 0
    attacker_cdata = CARD_DB.get(attacker_id)
    pierce = _is_pierce_attack(attack)

    if not pierce and target_cdata is not None and target_cdata.cardId == ROCK_INN_CRUSTLE_ID:
        if attacker_cdata is not None and attacker_cdata.ex:
            return 0  # Rock Inn: full block, weakness/resistance moot

    damage = base
    if target_cdata is not None and target_cdata.weakness is not None and attacker_cdata is not None:
        if int(attacker_cdata.energyType) == int(target_cdata.weakness):
            damage *= 2
    if target_cdata is not None and target_cdata.resistance is not None and attacker_cdata is not None:
        if int(attacker_cdata.energyType) == int(target_cdata.resistance):
            reduction = deck_tables.get("resistance_reduction", _DEFAULT_RESISTANCE_REDUCTION)
            damage = max(0, damage - reduction)

    if not pierce:
        stadium_ids = {c.id for c in (obs.current.stadium or [])}
        if FULL_METAL_LAB_ID in stadium_ids and target_cdata is not None:
            if int(target_cdata.energyType) == int(EnergyType.METAL):
                damage = max(0, damage - 30)

    return max(0, damage)


def _energy_deficit(attack_energies: list[int], attached_energies: list[int]) -> dict[int, int] | None:
    """How much more energy (by type) the attacker still needs to pay `attack_energies`,
    given what's already attached. COLORLESS cost slots can be paid by ANY leftover energy
    after specific-type slots are satisfied. A RAINBOW-type attached energy is treated as a
    wildcard covering any single slot (specific or colorless) — this ignores rarer special-
    energy effects (extra cost reduction, discard-on-use, etc.), which is a deliberate
    simplification for a deck-agnostic core. Returns None if the attacker can pay right now.
    """
    COLORLESS = int(EnergyType.COLORLESS)
    RAINBOW = int(EnergyType.RAINBOW)
    req = Counter(int(e) for e in attack_energies)
    req_colorless = req.pop(COLORLESS, 0)
    avail = Counter(int(e) for e in attached_energies)
    wildcards = avail.pop(RAINBOW, 0)

    deficit: dict[int, int] = {}
    for etype, need in req.items():
        have = avail.get(etype, 0)
        used = min(have, need)
        avail[etype] = have - used
        short = need - used
        if short > 0 and wildcards > 0:
            used2 = min(wildcards, short)
            wildcards -= used2
            short -= used2
        if short > 0:
            deficit[etype] = short

    if req_colorless > 0:
        leftover = sum(avail.values()) + wildcards
        short_c = max(0, req_colorless - leftover)
        if short_c > 0:
            deficit[COLORLESS] = short_c

    return deficit or None


def _opponent_threat_damage(obs: Observation, defender: Pokemon, deck_tables: dict) -> int:
    """Max expected damage the OPPONENT's active Pokemon could deal to `defender` (one of
    OUR Pokemon) using an attack it can already pay for RIGHT NOW, or could pay for after
    ONE more energy attach on its next turn. Mirrors `needs_energy`/`_energy_deficit` but
    from the opponent's side, reading only publicly-visible board state (opponent's active
    card id + its already-attached `energies` array) — no hidden-info guessing. Reuses
    `_leaf_damage` (weakness/Rock-Inn/Full-Metal-Lab-aware) with attacker/defender roles
    reversed relative to our own offense math. Returns 0 if the opponent has no active, no
    known attacks, or none payable within one attach.

    Deliberately NOT a search: no multi-turn energy ramp, no accounting for a Trainer that
    could hand the opponent extra energy/damage, no modeling of them switching active. Just
    "what could their CURRENT active hit us for, next turn, with what's already on board" —
    the one-hop check this task calls for, not a minimax.
    """
    yi = obs.current.yourIndex
    opp = obs.current.players[1 - yi]
    if not opp.active or opp.active[0] is None:
        return 0
    opp_active = opp.active[0]
    opp_cdata = CARD_DB.get(opp_active.id)
    if opp_cdata is None:
        return 0
    defender_cdata = CARD_DB.get(defender.id)
    best = 0
    for atk_id in (opp_cdata.attacks or []):
        atk = ATTACK_DB.get(atk_id)
        if atk is None:
            continue
        deficit = _energy_deficit(atk.energies, opp_active.energies)
        if deficit and sum(deficit.values()) > 1:
            continue  # more than one attach away -- not a credible "next turn" threat yet
        dmg = _leaf_damage(obs, opp_active.id, atk, defender, defender_cdata, deck_tables)
        if dmg > best:
            best = dmg
    return best


def _retreat_option_available(obs: Observation) -> bool:
    """True iff this select actually offers a RETREAT option (used to gate the ATTACK-
    under-lethal-threat deprioritization below: no point discouraging the attack if there's
    nothing to switch to instead)."""
    opts = obs.select.option if obs.select else None
    return any(o.type == int(OptionType.RETREAT) for o in (opts or []))


def _score(expected_damage: int, is_lethal: bool, needs_energy: dict[int, int] | None,
           needs_evolution: bool, target_area: int, deck_tables: dict) -> float:
    score = float(expected_damage)
    if is_lethal:
        score += _LETHAL_BONUS
    if needs_energy:
        penalty = deck_tables.get("energy_step_penalty", _DEFAULT_ENERGY_STEP_PENALTY)
        score -= penalty * sum(needs_energy.values())
    if needs_evolution:
        score -= deck_tables.get("evolution_penalty", _DEFAULT_EVOLUTION_PENALTY)
    if target_area == int(AreaType.BENCH):
        score -= deck_tables.get("bench_target_penalty", _DEFAULT_BENCH_TARGET_PENALTY)
    return score


def _find_attack_index(obs: Observation, attack_id: int) -> int:
    if not obs.select or not obs.select.option:
        return -1
    for i, opt in enumerate(obs.select.option):
        if opt.type == int(OptionType.ATTACK) and opt.attackId == attack_id:
            return i
    return -1


def _candidates_for_attacker(obs: Observation, attacker_area: int, attacker_slot: int,
                              attacker_pokemon: Pokemon, attacker_id: int,
                              target_area: int, target_slot: int, target: Pokemon,
                              needs_evolution: bool, deck_tables: dict) -> list[Plan]:
    """One Plan candidate per attack the (possibly hypothetically-evolved) attacker_id
    knows, against `target`."""
    out: list[Plan] = []
    cdata = CARD_DB.get(attacker_id)
    if cdata is None:
        return out
    target_cdata = CARD_DB.get(target.id)
    for atk_id in (cdata.attacks or []):
        atk = ATTACK_DB.get(atk_id)
        if atk is None:
            continue
        dmg = _leaf_damage(obs, attacker_id, atk, target, target_cdata, deck_tables)
        deficit = _energy_deficit(atk.energies, attacker_pokemon.energies)

        extra_score = 0.0
        guards = deck_tables.get("matchup_guards") or []
        if guards:
            ctx = GuardContext(
                obs=obs, attacker_area=attacker_area, attacker_slot=attacker_slot,
                attacker_id=attacker_id, attacker=attacker_pokemon, attack_id=atk_id,
                base_damage=dmg, target_area=target_area, target_slot=target_slot,
                target=target, target_cdata=target_cdata, needs_energy=deficit,
                needs_evolution=needs_evolution,
            )
            extra_dmg = 0
            for guard in guards:
                try:
                    d, s = guard(ctx)
                    extra_dmg += d
                    extra_score += s
                except Exception as e:
                    _exc.skip("matchup_guard", e)
            dmg = max(0, dmg + extra_dmg)

        is_lethal = dmg > 0 and dmg >= target.hp
        score = _score(dmg, is_lethal, deficit, needs_evolution, target_area, deck_tables) + extra_score
        out.append(Plan(
            attacker_area=attacker_area, attacker_slot=attacker_slot, attacker_id=attacker_id,
            attack_id=atk_id, attack_index_hint=_find_attack_index(obs, atk_id) if not needs_evolution and not deficit else -1,
            target_area=target_area, target_slot=target_slot, target_id=target.id,
            expected_damage=dmg, is_lethal=is_lethal,
            needs_energy=deficit, needs_evolution=needs_evolution, score=score,
        ))
    return out


# ── build_plan ──────────────────────────────────────────────────────────────────────────
def build_plan(obs: Observation, deck_tables: dict | None = None) -> Plan | None:
    """Enumerate attacker x attack x target and return the single best Plan (or None if no
    attacker/attack/target combination exists at all — e.g. we have no in-play Pokemon with
    any known attack). See module docstring for the deck_tables mechanism.

    attacker candidates: every one of our in-play Pokemon (active + bench), in its CURRENT
    form, PLUS — for each hand card that is a valid evolution of that Pokemon (CardData.
    evolvesFrom == the board Pokemon's CardData.name) — a hypothetical post-evolution form
    (needs_evolution=True), using the same attached energies (evolving doesn't change them).

    target candidates: the opponent's Active (always a legal attack target) plus, only if
    deck_tables["consider_bench_snipe"] is truthy, the opponent's bench (reachable only via
    a Boss's-Orders-style gust play — see serves_plan's PLAY branch).
    """
    tables = _tables(deck_tables)
    try:
        me = obs.current.players[obs.current.yourIndex]
        opp = obs.current.players[1 - obs.current.yourIndex]

        attackers: list[tuple[int, int, Pokemon]] = []
        if me.active and me.active[0] is not None:
            attackers.append((int(AreaType.ACTIVE), 0, me.active[0]))
        for i, p in enumerate(me.bench or []):
            if p is not None:
                attackers.append((int(AreaType.BENCH), i, p))

        targets: list[tuple[int, int, Pokemon]] = []
        if opp.active and opp.active[0] is not None:
            targets.append((int(AreaType.ACTIVE), 0, opp.active[0]))
        if tables.get("consider_bench_snipe"):
            for i, p in enumerate(opp.bench or []):
                if p is not None:
                    targets.append((int(AreaType.BENCH), i, p))

        hand = me.hand or []
        candidates: list[Plan] = []
        for area, slot, pokemon in attackers:
            cur_cdata = CARD_DB.get(pokemon.id)
            for t_area, t_slot, target in targets:
                candidates.extend(_candidates_for_attacker(
                    obs, area, slot, pokemon, pokemon.id, t_area, t_slot, target, False, tables))
                if cur_cdata is not None:
                    for hc in hand:
                        hc_data = CARD_DB.get(hc.id)
                        if hc_data is not None and hc_data.evolvesFrom == cur_cdata.name:
                            candidates.extend(_candidates_for_attacker(
                                obs, area, slot, pokemon, hc.id, t_area, t_slot, target, True, tables))

        if not candidates:
            STATS["plan_missing"] += 1
            return None
        best = max(candidates, key=lambda p: p.score)
        STATS["plan_built"] += 1
        return best
    except Exception as e:
        _exc.skip("build_plan", e)
        return None


def _plan_still_valid(obs: Observation, plan: Plan) -> bool:
    """attacker/target may have died, been replaced, or (for needs_evolution plans)
    evolved in-place since the plan was built — all handled here without forcing a rebuild
    when the change is simply "the plan is progressing as intended"."""
    try:
        me = obs.current.players[obs.current.yourIndex]
        opp = obs.current.players[1 - obs.current.yourIndex]
        attacker = _pokemon_at(me, plan.attacker_area, plan.attacker_slot)
        target = _pokemon_at(opp, plan.target_area, plan.target_slot)
        if attacker is None or target is None:
            return False
        if target.id != plan.target_id or target.hp <= 0:
            return False
        if attacker.id == plan.attacker_id:
            return True  # ready, or already evolved into the planned attacker
        if plan.needs_evolution:
            cur = CARD_DB.get(attacker.id)
            planned = CARD_DB.get(plan.attacker_id)
            if cur is not None and planned is not None and planned.evolvesFrom == cur.name:
                return True  # still pre-evolution, evolution still pending and still legal
        return False
    except Exception as e:
        _exc.skip("plan_still_valid", e)
        return False


def _refresh_plan(obs: Observation, plan: Plan, deck_tables: dict) -> Plan | None:
    """Recompute needs_energy/needs_evolution/expected_damage/is_lethal/score for the SAME
    committed (attacker, attack, target) triple against the current board — keeps the plan
    numerically accurate turn-to-turn (energy gets attached, evolution happens) without
    letting the committed choice itself drift."""
    try:
        me = obs.current.players[obs.current.yourIndex]
        opp = obs.current.players[1 - obs.current.yourIndex]
        attacker = _pokemon_at(me, plan.attacker_area, plan.attacker_slot)
        target = _pokemon_at(opp, plan.target_area, plan.target_slot)
        if attacker is None or target is None:
            return None
        needs_evolution = attacker.id != plan.attacker_id
        for c in _candidates_for_attacker(
                obs, plan.attacker_area, plan.attacker_slot, attacker, plan.attacker_id,
                plan.target_area, plan.target_slot, target, needs_evolution, deck_tables):
            if c.attack_id == plan.attack_id:
                return c
        return None
    except Exception as e:
        _exc.skip("refresh_plan", e)
        return None


# ── serves_plan ─────────────────────────────────────────────────────────────────────────
def serves_plan(obs: Observation, opt: Option, plan: Plan,
                 deck_tables: dict | None = None) -> tuple[float, str]:
    """Single source of truth for how much a MAIN-select option (ATTACK/ATTACH/EVOLVE/
    RETREAT/PLAY) — or a CARD-select follow-up to RETREAT/Boss's Orders (SWITCH / TO_ACTIVE
    style contexts) — advances the committed `plan`. Returns (score_bonus, reason).
    ATTACH is checked against plan.needs_energy FIRST, before any other generic logic, per
    the task spec."""
    if plan is None:
        return 0.0, "no plan"
    tables = _tables(deck_tables)
    yi = obs.current.yourIndex
    opp_i = 1 - yi
    ot = opt.type

    if ot == int(OptionType.ATTACK):
        if opt.attackId == plan.attack_id and not plan.needs_energy and not plan.needs_evolution:
            if not plan.is_lethal and _retreat_option_available(obs):
                our_active = obs.current.players[yi].active[0] if obs.current.players[yi].active else None
                if our_active is not None and our_active.hp > 0:
                    threat = _opponent_threat_damage(obs, our_active, tables)
                    if threat >= our_active.hp:
                        bonus = tables.get(
                            "attack_under_lethal_threat_bonus", _DEFAULT_ATTACK_UNDER_LETHAL_THREAT_BONUS)
                        return bonus, (
                            f"execute planned attack (deprioritized: non-lethal and opponent can "
                            f"KO our active next turn -- threat={threat} >= hp={our_active.hp} -- "
                            f"retreating instead)")
            return 1_000_000.0, "execute planned attack"
        return 0.0, "attack: not (yet) the committed attack"

    if ot == int(OptionType.ATTACH):
        card = _option_played_card(obs, opt, yi)
        cdata = CARD_DB.get(card.id) if card else None
        if cdata is None or cdata.cardType not in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
            return 0.0, "attach: not an energy card"
        if opt.inPlayArea != plan.attacker_area or opt.inPlayIndex != plan.attacker_slot:
            return 0.0, "attach: target isn't the planned attacker"
        if not plan.needs_energy:
            return 0.0, "attach: planned attacker already has enough energy"
        etype = int(cdata.energyType)
        if plan.needs_energy.get(etype, 0) > 0:
            return 5000.0, f"attach: fills a specific need (type={etype})"
        if plan.needs_energy.get(int(EnergyType.COLORLESS), 0) > 0:
            return 800.0, "attach: pays the remaining colorless need"
        return 0.0, "attach: this energy type isn't needed by the plan"

    if ot == int(OptionType.EVOLVE):
        if not plan.needs_evolution:
            return 0.0, "evolve: plan doesn't require evolving"
        if opt.inPlayArea != plan.attacker_area or opt.inPlayIndex != plan.attacker_slot:
            return 0.0, "evolve: wrong board slot"
        card = _option_played_card(obs, opt, yi)
        if card is None or card.id != plan.attacker_id:
            return 0.0, "evolve: wrong card"
        return 5000.0, "evolve: turns current mon into the planned attacker"

    if ot == int(OptionType.RETREAT):
        if plan.attacker_area == int(AreaType.BENCH):
            return 3000.0, "retreat: make room to bring the planned bench attacker up"
        # Defensive check: our plan's attacker IS the active, so "serves the committed
        # attack" alone says 0 here -- but that's exactly the wear-down-death failure mode
        # (see task brief): a plan that never reconsiders RETREAT purely offensively lets
        # its active get chipped down and KO'd on the opponent's next hit instead of
        # ducking out. One-hop, publicly-visible-only threat check (no search): if the
        # opponent's active can already kill (or badly hurt) OUR active next turn with what
        # it has on board right now (or one attach away), retreating earns a real bonus
        # here even though it doesn't advance the committed attack.
        our_active = obs.current.players[yi].active[0] if obs.current.players[yi].active else None
        if our_active is not None and our_active.hp > 0:
            threat = _opponent_threat_damage(obs, our_active, tables)
            if threat >= our_active.hp:
                bonus = tables.get("retreat_lethal_threat_bonus", _DEFAULT_RETREAT_LETHAL_THREAT_BONUS)
                return bonus, f"retreat: survive -- opponent can KO our active next turn (threat={threat} >= hp={our_active.hp})"
            if threat * 2 >= our_active.hp:  # threat is >=50% of our current HP
                bonus = tables.get("retreat_heavy_threat_bonus", _DEFAULT_RETREAT_HEAVY_THREAT_BONUS)
                return bonus, f"retreat: opponent threatens a heavy hit next turn (threat={threat}, hp={our_active.hp})"
        return 0.0, "retreat: planned attacker is already active"

    if ot == int(OptionType.PLAY):
        card = _option_played_card(obs, opt, yi)
        cid = card.id if card is not None else opt.cardId
        if cid == BOSS_ORDERS_CARD_ID and plan.target_area == int(AreaType.BENCH):
            return 4000.0, "play Boss's Orders: only way to reach the planned bench target"
        return 0.0, "play: unrelated to the plan"

    if ot == int(OptionType.CARD):
        # Follow-up CARD select after RETREAT (SWITCH-style context) or after playing
        # Boss's Orders (TO_ACTIVE-style context): both just need "pick our planned
        # attacker's bench slot" / "pick the opponent's planned-target bench slot" — same
        # rule regardless of the exact SelectContext value the engine happens to use.
        pi = opt.playerIndex if opt.playerIndex is not None else yi
        if pi == yi and opt.area == plan.attacker_area and opt.index == plan.attacker_slot:
            return 3000.0, "card: brings the planned attacker into the active spot"
        if pi == opp_i and opt.area == plan.target_area and opt.index == plan.target_slot:
            return 3000.0, "card: forces the planned target into the active spot"
        return 0.0, "card: unrelated to the plan"

    return 0.0, "no serves_plan rule for this option type"


# ── main entry point ────────────────────────────────────────────────────────────────────
_plan: Plan | None = None
_loop_sig: str | None = None
_loop_repeat: int = 0


def reset_state() -> None:
    """Reset ALL module-level per-episode state. Called on every obs["select"] is None
    (new-episode boundary), per the project-wide reset convention (canonical example:
    submissions/v076_final/main.py:1528-1544)."""
    global _plan, _loop_sig, _loop_repeat
    _plan = None
    _loop_sig = None
    _loop_repeat = 0


def _pick_k(min_count: int, max_count: int, n: int) -> int:
    return max(1, min(min_count, max_count, n))


def _fallback_pick(obs_dict: dict, sel_dict: dict | None) -> list[int]:
    STATS["generic_fallback"] += 1
    try:
        pick = greedy_meta_agent(obs_dict)
    except Exception as e:
        _exc.skip("greedy_meta_agent", e)
        pick = None
    if pick:
        return pick
    opts = (sel_dict or {}).get("option") or []
    n = len(opts)
    if n == 0:
        return []
    k = _pick_k(int((sel_dict or {}).get("minCount") or 1), int((sel_dict or {}).get("maxCount") or 1), n)
    return list(range(k))


def choose_options(obs_dict: dict, deck_tables: dict | None = None) -> list[int]:
    """Public per-select entry point. Submission wrappers should call this directly with
    the raw obs dict the engine hands them, plus optionally their own deck_tables (or call
    set_deck_tables() once at startup and omit it here).

    1. obs_dict["select"] is None -> reset state, answer with deck_tables["deck"] (60 ids).
    2. Loop-breaker (same select signature repeating) takes priority over everything else.
    3. On a MAIN select, (re)build/refresh the committed plan if missing/invalid.
    4. If no plan exists at all, fall back to greedy_meta_agent for this whole step.
    5. Otherwise score every option via serves_plan + a small generic type-priority, and
       return the top-k indices.
    """
    global _plan, _loop_sig, _loop_repeat
    tables = _tables(deck_tables)

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        reset_state()
        deck = tables.get("deck")
        if not deck:
            _exc.skip("missing_deck_tables_deck")
            return []
        return list(deck)

    try:
        obs = to_observation_class(obs_dict)
    except Exception as e:
        _exc.skip("to_observation_class", e)
        return _fallback_pick(obs_dict, sel)

    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return []
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    context = sel.get("context")

    # Loop-breaker (ported pattern from submissions/v076_final/main.py:1558-1574).
    sig = _select_signature(obs_dict)
    if sig is not None and sig == _loop_sig:
        _loop_repeat += 1
    else:
        _loop_sig = sig
        _loop_repeat = 0
    if _loop_repeat >= 2:
        shift = (_loop_repeat - 1) % n
        k = _pick_k(min_count, max_count, n)
        pick = [(i + shift) % n for i in range(k)]
        _log(f"loop_breaker repeat={_loop_repeat} shift={shift} pick={pick}")
        return pick

    if context == int(SelectContext.MAIN):
        if _plan is None or not _plan_still_valid(obs, _plan):
            _plan = build_plan(obs, tables)
        else:
            refreshed = _refresh_plan(obs, _plan, tables)
            _plan = refreshed if refreshed is not None else build_plan(obs, tables)
        if _plan is not None:
            _log(f"turn={obs.current.turn} plan attacker={_plan.attacker_id}@{_plan.attacker_area}/{_plan.attacker_slot} "
                 f"attack={_plan.attack_id} target={_plan.target_id} dmg={_plan.expected_damage} "
                 f"lethal={_plan.is_lethal} needs_energy={_plan.needs_energy} needs_evolution={_plan.needs_evolution}")

    if _plan is None:
        return _fallback_pick(obs_dict, sel)

    scored: list[tuple[float, float, int, str]] = []  # (total, bonus, index, reason)
    for i, opt in enumerate(obs.select.option):
        try:
            bonus, reason = serves_plan(obs, opt, _plan, tables)
        except Exception as e:
            _exc.skip("serves_plan", e)
            bonus, reason = 0.0, "exception"
        base = _OPTION_TYPE_PRIORITY.get(opt.type, 1.0)
        scored.append((base + bonus, bonus, i, reason))

    scored.sort(key=lambda t: -t[0])
    k = _pick_k(min_count, max_count, n)
    pick = [i for _, _, i, _ in scored[:k]]

    top_total, top_bonus, top_i, _ = scored[0]
    if top_bonus > 0:
        try:
            _bump_served(OptionType(options[top_i].get("type") if isinstance(options[top_i], dict) else obs.select.option[top_i].type).name)
        except Exception:
            pass
    runner_up_total = scored[k][0] if len(scored) > k else (scored[-1][0] if len(scored) > 1 else top_total)
    margin = top_total - runner_up_total
    ctx_name = context
    try:
        ctx_name = SelectContext(context).name
    except Exception:
        pass
    _log(f"ctx={ctx_name} served_plan={top_bonus > 0} pick={pick} margin={margin}")

    return pick
