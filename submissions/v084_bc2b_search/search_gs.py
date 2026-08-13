"""[v084 v2] search-override module: playout (Monte-Carlo forward simulation to a fixed
turn-handover depth) + trained-value-model leaf, on top of the BC-priority ranker's root
candidates. Replaces v1's 2-ply determinized minimax + linear leaf, which the full gate
(PLAN_v084_SEARCH.md §3bis) REFUTED: band_700_799 -7.28pp (z=-5.29), band_800_plus -9.37pp
(z=-5.53), 12582 overrides (2.9% of decisions) traced to a systematically biased leaf
evaluation, not to variance -- margin-gating could not have saved that design.

Architecture PORTED (scheme, not code -- PLAN_v084_SEARCH.md §3ter): the same idea proven
in this project's OWN prior work (`submissions/scout_deeper/main.py:426-544`'s playout(),
the only design in this repo's whole search history that passed a full gate, +5.8pp
p=0.0002) -- BC top-K root candidates -> Monte-Carlo determinize hidden zones ->
search_begin a hidden-info world -> playout forward using each side's OWN policy (not a
generic heuristic for both) until a fixed number of turn-handovers -> evaluate the leaf
with a MODEL trained on real game outcomes, not a hand-tuned linear formula -> UCB1-driven
sample allocation across candidates -> margin-gated override of the BC pick. The CODE below
is this project's own: `_step_raw`/`_playout`/`_ucb1_pick`/`_pick_for_actor` are each
written with this module's own data structures and control flow (a single parameterized
playout loop dispatching to a per-actor policy function, not two near-duplicate own/
opponent rollout functions; a lazily-grown, cyclically-reused world list rather than
scout's upfront `roots = [...]`; UCB1 arm selection folded into one small pure function
instead of scout's inline branch) -- see each function's docstring for the specific
divergence from the reference. Never-throw: `decide()` catches everything internally and
always returns `(None | list[int], stats_dict)`; the caller (main.py) treats `None` as
"keep the BC pick" everywhere, unchanged from v1.

──────────────────────────────────────────────────────────────────────────
KEY UNLOCK -- raw obs_dict from inside a search rollout (v1 could not do this; see v1's own
removed docstring section, kept here in spirit): `lib.SearchStep` (cg/api.py:597-599) returns
a JSON byte string; the typed `search_step()` wrapper feeds it through `json_to_dataclass`
(cg/api.py:608, cg/utils.py) to get a typed `Observation`. `_step_raw` below skips that last
step and does `json.loads(bs.decode())["state"]["observation"]` directly -- proven by
`scripts/_tmp_v084_stepraw_probe.py` (run 2026-08-12, cg_native/arm64 stand) to produce a
dict with top-level keys `['current', 'logs', 'select']`, structurally identical to the
obs_dict main.py's `agent()` receives for a REAL decision, and on which
`bc_features_2b.build_features` runs successfully (272-length vector, no exception). This is
what makes running BC INSIDE the rollout (not just at the root, as v1 did) cheap and safe.

Root creation still goes through the TYPED `api.search_begin`, not a raw `_begin_raw`
counterpart. Decision, not oversight: (a) `search_begin`'s Python wrapper does real
validation (deck/prize/hand length checks against `agent_observation.current`,
cg/api.py:553-572) that is cheap insurance against a hidden-info sampling bug silently
producing a malformed world -- worth keeping since it runs once per world (up to N_DET times
per decision), not once per playout step; (b) the ROOT's own observation is never needed in
raw form -- the root's `Observation.observation` IS `obs_dict` itself (same decision the
caller already has), so there is nothing to gain from a raw root. `_step_raw` is used for
every `search_step` call AFTER the root (i.e. every step actually taken during a playout),
which is where the volume is and where the raw shortcut matters for BC-in-rollout.

──────────────────────────────────────────────────────────────────────────
RISK FOUND AND HANDLED -- shared mutable BC state across real vs. simulated calls:
`bc_features_2b.build_features` deliberately keeps one piece of state alive ACROSS calls at
module scope -- the turn-history buffer (`_turn_hist_turn/_turn_hist_buf/_turn_hist_last_sig/
_turn_hist_ability_used`, bc_features_2b.py:607-611), harvested from `obs['logs']` and
deduplicated by a signature of the calling decision (bc_features_2b.py:1063-1066's own
docstring flags this as the one intentional exception to "no cache between calls"). Both
main.py's real `_bc_decide`/`_bc_root_scores` AND this module's own BC-in-rollout calls
(`_pick_for_actor` below) resolve `import bc_features_2b` to the SAME module object (single
sys.path entry, single module-name cache -- confirmed empirically, see `_get_bc_api`'s
docstring) -- so calling `build_features` on a hypothetical rollout state and calling it on
the live root decision are indistinguishable to that module's internal state, and a rollout
that visits a same-numbered turn as the real game (routine, since playouts start from the
CURRENT real turn) would silently corrupt the buffer that the NEXT REAL decision this
episode reads. Fix: `decide()` snapshots those four globals immediately before the
playout/UCB1 loop and restores them in a `finally`, unconditionally (exception or not) --
see the snapshot/restore block in `decide()`. `scripts/_tmp_v084_stepraw_probe.py`'s test 5
exercises the snapshot/restore round-trip mechanically (its "no visible mutation" result
there is a signature-dedup artifact of probing on a single fixed obs, not a claim that the
underlying risk is absent -- the risk is established directly from bc_features_2b's own
source, and the fix does not depend on being able to force a visible mutation in a probe).

──────────────────────────────────────────────────────────────────────────
Value model -- PARALLEL TRACK, NOT YET LANDED (PLAN_v084_SEARCH.md §3ter item 3): this
module lazy-imports `value_features_gs`/`value_model_gs` (`_get_value_api`) and, per the
task's own instruction, treats their absence as "search unavailable" -- `decide()` returns
`(None, stats)` with `reason="no_value_model"` on every call, i.e. exact v083 parity, NOT a
silent fallback to v1's refuted linear leaf. The two-function contract
(`value_features(raw_obs, me_index) -> list[float]`, then `value_predict(feat) -> float` in
[0,1]) is this module's own specification for that future module, mirrored on
bc_features_2b/bc_model_2b's own features-then-model split in this same codebase, since no
value_*_gs.py exists yet to read a contract off of. `me_index` passed to `value_features` at
every call site is the LEAF's own `current.yourIndex` (own-hand features are only populated
from the mover's perspective in this snapshot -- value_features_gs.py's own docstring,
"ВИДИМОСТЬ" section -- exactly like bc_features_2b), NOT the search root's fixed side; `_value_leaf`
(below) then flips the resulting probability to the root's perspective when the leaf's mover
differs from the root. [v2 fixer, review finding 1] An earlier version of this module passed
the fixed root index directly and skipped the flip -- ported here verbatim from this
project's own gate-passed prior art, `submissions/scout_deeper/main.py:395-408`'s
`evaluate()`, after the mismatch was caught empirically (16.9% of real playout leaves land on
the non-root mover, mean prediction bias -0.079 when uncorrected).

Never-throw contract: `decide()` catches everything internally and always returns
`(None | list[int], stats_dict)`. `reset_episode()` cannot raise (pure dict resets).
Callers should still wrap the call in their own try/except as defense-in-depth (main.py
does), but nothing in normal operation should reach it.
"""
from __future__ import annotations

import ctypes
import json
import math
import random
import time
from collections import Counter

try:
    import matchup_decks_gs as _mdecks
    _TEMPLATES: dict = dict(_mdecks.TEMPLATES)
except Exception:  # pragma: no cover - defensive, matches project-wide pattern
    _TEMPLATES = {}

# ── Tunables (PLAN_v084_SEARCH.md §3ter) ─────────────────────────────────────
N_DET = 3                          # hidden-info worlds (search_begin calls), lazily created,
                                    # reused cyclically across playouts within one decision
K_CAND = 6                         # BC top-K candidates considered at the search root
DEPTH = 2                          # playout stops after this many turn-owner handovers
N_PLAYOUTS = 24                    # soft cap on (candidate, world) samples per decision --
                                    # the real limiter is TIME_BUDGET_S/deadline below; this
                                    # is only a backstop against a pathologically cheap playout
UCB_C = 0.2                        # UCB1 exploration constant (rewards are p in [0,1])
TIME_BUDGET_S = 0.8                # wall-clock budget per decision
DELTA_P = 0.03                     # override threshold, leaf-probability units (p in [0,1]) --
                                    # calibrated 2026-08-12 (fixer v2, 48+20-game smoke/mirror
                                    # trace): 3.94% of search-launched decisions override at this
                                    # threshold, centered in the healthy 2-10% zone and below the
                                    # v1-incident rate of 2.9% of ALL decide() calls
MAX_PLAYOUT_STEPS = 60             # cap on search_step calls within ONE playout
EPISODE_SOFT_S = 380.0             # cumulative episode search time -> degrade to N_DET=1
EPISODE_HARD_S = 520.0             # cumulative episode search time -> search off for rest of game
MAX_ENGINE_STEPS_PER_DECISION = 600  # hard cap on total search_step calls in one decide() call
MAX_SEARCH_EXCEPTIONS = 3          # cumulative-episode exception budget before permanent disable
MIN_OPTIONS = 3
MAX_OPTIONS = 24
MIN_TURN = 2

DUMMY_BASIC = 646                  # Impidimp (any basic Pokemon card ID) -- padding filler only
DUMMY_ENERGY = 7                   # Darkness basic energy -- padding filler only
_BASIC_ENERGY = {i: i for i in range(1, 9)}  # EnergyType 1-8 IDs mirror basic-energy card IDs 1-8

# [v1 fixer p.1, correctness review finding 1] Hand-scaling attacks whose engine-listed
# damage is 0/"n/a" because the real damage is computed at resolution time from the
# attacker's own hand size. Own local copy of the same fact main.py's
# _SCALING_ATTACKERS/scoring_grimmsnarl.py's SCALING_ATTACKERS encode (форензик v064
# БАГ-9, card 743 = Alakazam / Powerful Hand): value = (HP per card in attacker's
# hand, minimum energy required). A separate raw-dict constant here rather than an
# import from main.py -- this module must not depend on main.py (import direction is
# main.py -> search_gs.py, never the reverse) -- matching the established "two
# representations of the same engine fact" split already used elsewhere in this repo.
_SCALING_ATTACKERS = {743: (20, 1)}

# ── Episode-cumulative state (module-level; reset via reset_episode()) ──────
_SPENT = {
    "t": 0.0,           # cumulative wall-clock seconds spent inside decide() this episode
    "n_calls": 0,        # number of decide() invocations this episode
    "n_overrides": 0,    # number of times decide() returned a non-None override
    "n_degraded": 0,     # number of decisions run at reduced N_DET due to EPISODE_SOFT_S
    "n_exceptions": 0,   # cumulative-episode exception count (see decide()'s docstring)
    "disabled": False,   # permanently off for the rest of this episode
}

_api = None          # lazily-populated dict of cg.api symbols, cached after first success
_api_failed = False   # sticky: once the lazy import fails, don't retry every decision

_bc_api = None        # lazily-populated dict of bc_features_2b/bc_model_2b symbols
_bc_api_failed = False

_value_api = None      # lazily-populated dict of value_features_gs/value_model_gs symbols
_value_api_failed = False


def reset_episode() -> None:
    """Call once per new game (main.py's `sel is None` branch). Cannot raise."""
    _SPENT["t"] = 0.0
    _SPENT["n_calls"] = 0
    _SPENT["n_overrides"] = 0
    _SPENT["n_degraded"] = 0
    _SPENT["n_exceptions"] = 0
    _SPENT["disabled"] = False


def _get_api():
    """Lazy, cached import of the engine search API + attack table, PLUS (v2) the raw
    `cg.api` module object itself (`capi_mod`) and its ctypes `lib` handle -- needed by
    `_step_raw` to call `lib.SearchStep` directly and to read the module-global
    `agent_ptr` that the typed `search_begin` call below sets on first use. Returns None
    (and stays None for the rest of the process) on any failure -- the module import
    itself never touches cg.api at all, per PLAN_v084_SEARCH.md ("import cg.api —
    ленивый, внутри функции, под try/except")."""
    global _api, _api_failed
    if _api is not None:
        return _api
    if _api_failed:
        return None
    try:
        from cg.api import (
            to_observation_class, search_begin, search_step, search_end,
            search_release, SelectContext, all_attack,
        )
        import cg.api as _capi_mod
        attacks = {}
        for a in all_attack():
            try:
                attacks[int(a.attackId)] = a
            except Exception:
                continue
        _api = dict(
            to_observation_class=to_observation_class,
            search_begin=search_begin,
            search_step=search_step,   # kept for parity/back-compat; playouts use _step_raw
            search_end=search_end,
            search_release=search_release,
            SelectContext=SelectContext,
            attacks=attacks,
            capi_mod=_capi_mod,
            lib=_capi_mod.lib,
        )
        return _api
    except Exception:
        _api_failed = True
        return None


def _get_bc_api():
    """[v2] Lazy, cached import of THIS submission's own bc_features_2b/bc_model_2b,
    imported directly -- NOT routed through main.py (import direction stays main.py ->
    search_gs.py only). Because both this module and main.py resolve `import
    bc_features_2b` off the same sys.path entry (the submission directory), Python's
    module-name cache hands back the SAME module object main.py already imported as
    `_bcf` -- there is only ever one bc_features_2b module instance in the process. That
    is exactly why `decide()` must snapshot/restore that module's turn-history globals
    around every rollout (see this module's docstring, "RISK FOUND AND HANDLED" section)
    -- without it, BC-in-rollout calls here would corrupt state main.py's next REAL
    decision this episode depends on. On any import failure, BC-in-rollout is simply
    unavailable and `_pick_for_actor` always falls back to the generic scorer -- this is
    a soft dependency, unlike the value model (see `_get_value_api`)."""
    global _bc_api, _bc_api_failed
    if _bc_api is not None:
        return _bc_api
    if _bc_api_failed:
        return None
    try:
        import bc_features_2b as _bcf_mod
        import bc_model_2b as _bcm_mod
        _bc_api = dict(
            module=_bcf_mod,
            route_group=_bcf_mod.route_group,
            build_features=_bcf_mod.build_features,
            predict_group=_bcm_mod.predict_group,
        )
        return _bc_api
    except Exception:
        _bc_api_failed = True
        return None


def _get_value_api():
    """[v2] Lazy, cached import of value_features_gs/value_model_gs -- a PARALLEL,
    not-yet-landed track (PLAN_v084_SEARCH.md §3ter item 3). While these files don't
    exist, search is UNAVAILABLE, full stop: no fallback to v1's linear leaf (that design
    was refuted by the full gate -- resurrecting it as a silent fallback would silently
    re-introduce the exact defect the gate caught, undetectably to anyone reading a
    `src=bc` trace line thinking search is simply declining to override). See this
    module's own docstring for the assumed two-call contract these files must implement."""
    global _value_api, _value_api_failed
    if _value_api is not None:
        return _value_api
    if _value_api_failed:
        return None
    try:
        import value_features_gs as _vf_mod
        import value_model_gs as _vm_mod
        vf = _vf_mod.value_features
        vp = _vm_mod.value_predict
        _value_api = dict(value_features=vf, value_predict=vp)
        return _value_api
    except Exception:
        _value_api_failed = True
        return None


# ── Generic in-rollout option scorer, raw-dict flavor (own implementation) ──────────────
# Priority ordering per PLAN_v084_SEARCH.md §2.2 step 6's non-mirror branch (kept from v1
# unchanged): ATTACK (by damage) > ABILITY > EVOLVE/ATTACH/PLAY > ENERGY-ish > CARD-ish >
# DISCARD > RETREAT > YES/NO > END. Values are option-type codes from cg/api.py:120-187
# (OptionType IntEnum) -- read as plain ints off `opt["type"]` rather than importing the
# enum, to keep this module's cg.api surface area minimal. v2 operates on raw dicts
# throughout the rollout (see module docstring's "KEY UNLOCK" section), so this is a
# dict-keyed rewrite of v1's dataclass-attribute version -- same table, same logic.
_TYPE_PRIORITY = {
    13: 600,   # ATTACK  (damage added on top, see _option_score_raw)
    10: 500,   # ABILITY
    9: 400,    # EVOLVE
    8: 350,    # ATTACH
    7: 340,    # PLAY
    6: 300,    # ENERGY
    5: 290,    # ENERGY_CARD
    4: 280,    # TOOL_CARD
    3: 250,    # CARD
    15: 200,   # SKILL
    16: 190,   # SPECIAL_CONDITION
    0: 150,    # NUMBER
    11: 100,   # DISCARD
    12: 50,    # RETREAT
    1: 40,     # YES
    2: 30,     # NO
    14: 0,     # END
}


def _actor_context_raw(raw_obs: dict, actor: int) -> tuple:
    """(active-Pokemon card id, hand size) for whichever player `actor` is, read straight
    off a raw obs_dict's `current.players[actor]` -- used only to price hand-scaling
    attacks (see _SCALING_ATTACKERS). handCount is public for both sides; the active
    mon's identity is public once it's in play, so this needs no hidden-info assumption
    beyond what _sample_hidden already committed to for this branch."""
    cur = raw_obs.get("current") or {}
    players = cur.get("players") or []
    if not (0 <= actor < len(players)):
        return None, 0
    pl = players[actor] or {}
    active = pl.get("active") or []
    active_id = None
    if active and active[0] is not None:
        active_id = active[0].get("id")
    return active_id, int(pl.get("handCount") or 0)


def _option_score_raw(opt: dict, api, active_id, hand_size: int) -> float:
    t = int(opt.get("type") or 0)
    base = float(_TYPE_PRIORITY.get(t, 120))
    if t == 13:  # ATTACK: prefer the highest-damage attack available
        attack_id = opt.get("attackId")
        atk = api["attacks"].get(int(attack_id)) if attack_id is not None else None
        dmg = atk.damage if (atk is not None and isinstance(atk.damage, (int, float))) else 0
        scale = _SCALING_ATTACKERS.get(active_id) if active_id is not None else None
        if scale is not None:
            per_card, _min_energy = scale
            dmg = max(dmg, per_card * hand_size)
        base += float(dmg)
    return base


def _greedy_pick_raw(raw_obs: dict, actor: int, api) -> list:
    """Generic-scorer pick for whatever select `raw_obs['select']` presents. k = minCount
    if minCount>0 else min(maxCount, n) -- same convention _bc_decide/v1's _greedy_pick
    use, translated to raw dict access."""
    sel = raw_obs.get("select") or {}
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return []
    active_id, hand_size = _actor_context_raw(raw_obs, actor)
    scores = [_option_score_raw(o or {}, api, active_id, hand_size) for o in options]
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    min_c = int(sel.get("minCount") or 0)
    max_c = int(sel.get("maxCount") or 1)
    max_c = max(max_c, 1)
    k = min_c if min_c > 0 else min(max_c, n)
    k = max(1, min(k, max_c, n))
    return order[:k]


def _pick_for_actor(raw_obs: dict, actor: int, me_i: int, matchup: str, bc_api, api) -> list:
    """[v2] Per-step rollout policy dispatch (PLAN_v084_SEARCH.md §3ter item 1): our own
    side always tries BC first; the opponent's side tries BC only in the mirror matchup
    (they are playing OUR OWN deck there, so our BC is a legitimate model of their play
    too). Falls back to the local generic scorer (`_greedy_pick_raw`) whenever: BC is
    unavailable (`bc_api is None`), this isn't a plain single-pick decision
    (minCount==maxCount==1 -- multi-select sub-decisions like discard/energy-target
    selection are out of BC's live routing contract, same rule `_bc_decide` uses via its
    k-selection logic), or BC's own routing abstains (`route_group` returns None/'rest',
    or scoring raises for any reason)."""
    sel = raw_obs.get("select")
    if sel is None:
        return []
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return []
    min_c = int(sel.get("minCount") or 0)
    max_c = int(sel.get("maxCount") or 1)
    try_bc = (bc_api is not None and min_c == 1 and max_c == 1
              and (actor == me_i or matchup == "mirror"))
    if try_bc:
        ctx = sel.get("context")
        try:
            group = bc_api["route_group"](ctx if ctx is not None else -1)
        except Exception:
            group = None
        if group is not None and group != "rest":
            try:
                scores = [bc_api["predict_group"](group, bc_api["build_features"](raw_obs, opt, i, n))
                          for i, opt in enumerate(options)]
                order = sorted(range(n), key=lambda i: (-scores[i], i))
                return [order[0]]
            except Exception:
                pass  # fall through to generic below
    return _greedy_pick_raw(raw_obs, actor, api)


# ── Hidden-info determinization (UNCHANGED from v1 -- own implementation of the scheme
# described in PLAN_v084_SEARCH.md §2.1 step 3: sample each side's unknown deck/prize/hand
# cards as a uniform draw from "declared 60-list minus everything already visible", falling
# back to a synthetic pool when no archetype template matches. Operates on the TYPED root
# Observation/State once per world (not per rollout step), so there is no reason to port it
# to raw-dict access the way the rollout's own option scorer was. ─────────────────────────

def _zone_multiset(pl, include_hand: bool) -> Counter:
    """Card-ID multiset of everything about player-state `pl` whose identity is public
    knowledge: discard pile, face-up prizes, and the whole board (active + bench, plus
    each mon's attached energy/tool/pre-evolution stack). `include_hand` additionally
    counts `pl.hand` -- populated by the engine only for the searching side, never for
    the opponent (cg/api.py:359), so callers pass True for "me" and False for "them"."""
    counts: Counter = Counter()
    for c in pl.discard:
        counts[c.id] += 1
    if include_hand:
        for c in (pl.hand or []):
            counts[c.id] += 1
    for c in pl.prize:
        if c is not None:
            counts[c.id] += 1
    for mon in list(pl.active or []) + list(pl.bench or []):
        if mon is None:
            continue
        counts[mon.id] += 1
        for c in mon.energyCards:
            counts[c.id] += 1
        for c in mon.tools:
            counts[c.id] += 1
        for c in mon.preEvolution:
            counts[c.id] += 1
    return counts


def _energy_type_histogram(pl) -> Counter:
    """EnergyType tally across `pl`'s whole board -- used only to guess a basic-energy id
    when no archetype template matches (synthesis fallback)."""
    hist: Counter = Counter()
    for mon in list(pl.active or []) + list(pl.bench or []):
        if mon is None:
            continue
        for e in mon.energies:
            hist[int(e)] += 1
    return hist


def _subtract_counts(template: Counter, seen: Counter) -> list:
    """Multiset difference template - seen, flattened to a plain list. Counter's `-`
    operator already clamps any key that would go negative to zero, so no manual
    per-key max(0, n - seen.get(cid, 0)) bookkeeping is needed here."""
    return list((template - seen).elements())


def _synthesize_opponent_pool(seen_opp: Counter, energy_hist: Counter) -> list:
    """No archetype template matched this opponent (unrecognized/rogue deck): approximate
    their hidden zones with their single most-seen visible card plus basic energy of
    their most-attached type, sized per PLAN_v084_SEARCH.md §2.1 step 3's explicit
    fallback spec (30/30/8 -- these three numbers are the plan's own spec, not
    implementation detail)."""
    dominant_energy = max(energy_hist.items(), key=lambda kv: kv[1])[0] if energy_hist else 7
    energy_id = _BASIC_ENERGY.get(dominant_energy, 7)
    counts: Counter = Counter()
    if seen_opp:
        top_card = max(seen_opp.items(), key=lambda kv: kv[1])[0]
        counts[top_card] += 30
    counts[energy_id] += 30
    counts[DUMMY_BASIC] += 8
    return list(counts.elements())


def _pad_to(pool: list, target_len: int, filler) -> list:
    if len(pool) < target_len:
        return pool + [filler] * (target_len - len(pool))
    return pool


def _carve(pool: list, *sizes: int) -> list:
    """Slice `pool` into consecutive, non-overlapping chunks of the given sizes, in
    order; any leftover past sum(sizes) is dropped. Caller guarantees
    len(pool) >= sum(sizes) (via _pad_to)."""
    chunks = []
    start = 0
    for size in sizes:
        chunks.append(pool[start:start + size])
        start += size
    return chunks


def _fill_facedown(cards, filler_chunk: list) -> list:
    """Card-id list matching `cards` 1:1: known cards keep their id, facedown (None)
    slots take the next id from `filler_chunk` in order."""
    out = []
    j = 0
    for c in cards:
        if c is not None:
            out.append(c.id)
        else:
            out.append(filler_chunk[j] if j < len(filler_chunk) else DUMMY_ENERGY)
            j += 1
    return out


def _sample_hidden(state, me_i: int, my_deck: list, opp_template) -> dict:
    """One determinization of all hidden zones -> kwargs for search_begin. `opp_template`:
    list[int] (60 card IDs, a specific archetype hypothesis -- including our own my_deck
    for the mirror matchup) or None (no matchup detected -> synthesize, see
    _synthesize_opponent_pool)."""
    me = state.players[me_i]
    op = state.players[1 - me_i]

    seen_mine = _zone_multiset(me, include_hand=True)
    seen_opp = _zone_multiset(op, include_hand=False)
    opp_energy_hist = _energy_type_histogram(op)
    for c in (state.stadium or []):
        (seen_mine if c.playerIndex == me_i else seen_opp)[c.id] += 1

    my_hidden = _subtract_counts(Counter(my_deck), seen_mine)
    n_my_prize_hidden = sum(1 for c in me.prize if c is None)
    my_hidden = _pad_to(my_hidden, me.deckCount + n_my_prize_hidden, DUMMY_ENERGY)
    random.shuffle(my_hidden)
    your_deck, my_prize_filler = _carve(my_hidden, me.deckCount, n_my_prize_hidden)
    your_prize = _fill_facedown(me.prize, my_prize_filler)

    if opp_template:
        opp_hidden = _subtract_counts(Counter(opp_template), seen_opp)
    else:
        opp_hidden = _synthesize_opponent_pool(seen_opp, opp_energy_hist)

    n_opp_prize_hidden = sum(1 for c in op.prize if c is None)
    opp_need = op.deckCount + n_opp_prize_hidden + op.handCount
    opp_hidden = _pad_to(opp_hidden, opp_need, DUMMY_ENERGY)
    random.shuffle(opp_hidden)
    opponent_deck, opp_prize_filler, opponent_hand = _carve(
        opp_hidden, op.deckCount, n_opp_prize_hidden, op.handCount)
    opponent_prize = _fill_facedown(op.prize, opp_prize_filler)
    opponent_active = [DUMMY_BASIC] if (op.active and op.active[0] is None) else []

    return dict(
        your_deck=your_deck, your_prize=your_prize,
        opponent_deck=opponent_deck, opponent_prize=opponent_prize,
        opponent_hand=opponent_hand, opponent_active=opponent_active,
    )


# ── Leaf evaluation: trained value model + exact terminal result (REPLACES v1's linear
# formula, refuted by the full gate -- see module docstring). ───────────────────────────

def _terminal_value(result: int, me_i: int) -> float:
    """result==me -> 1.0, draw -> 0.5, else -> 0.0 (PLAN_v084_SEARCH.md §3ter item 3).
    `result==2` as the draw code is this project's own established reading of the
    engine's State.result field (cg/api.py:376 only documents "win player index, -1 if
    unfinished") -- v1's `_leaf_eval` already used this convention and ran clean through
    the full 4400-game gate, so it is kept rather than re-derived here."""
    if result == me_i:
        return 1.0
    if result == 2:
        return 0.5
    return 0.0


def _value_leaf(raw_obs: dict, me_i: int, value_api) -> float:
    """p(me wins) in [0,1] from the trained value model.

    [v2 fixer, review finding 1 -- SHIP-BLOCKER, fixed] Own-hand features
    (`own_*_n`, value_features_gs.py:258-262) are visible ONLY from the perspective of
    whichever side is `current.yourIndex` IN THIS SNAPSHOT (value_features_gs.py's own
    docstring, "ВИДИМОСТЬ" section, lines 34-42) -- exactly like bc_features_2b, and
    exactly like scout_deeper/main.py's `evaluate()` (submissions/scout_deeper/main.py:
    395-408) already handles for this project's OWN prior, gate-passed value-model
    integration. The leaf reached mid-playout can land on EITHER side's turn (module
    docstring's own comment: "current.yourIndex flips turn to turn"), so calling
    `value_features(raw_obs, me_i)` with the fixed SEARCH-ROOT `me_i` -- as this function
    used to -- feeds the model a snapshot whose own-hand fields are SENTINEL (-1.0)
    whenever the leaf's mover differs from the root's `me_i` (measured empirically at
    16.9% of real playout leaves; mean prediction bias -0.079, non-random -- see fixer
    review). Training data never sees that SENTINEL (build_value_dataset_gs.py always
    calls `value_features(obs, agent_index)` with `agent_index == yourIndex` of that
    snapshot), so this was a train/serve skew invisible to offline AUC/calibration.
    Fix, ported verbatim from scout_deeper's `evaluate()`: compute features from the
    LEAF's own mover perspective (features are always "eyes of whoever is about to
    act" -- fully populated, no sentinel), get p(mover wins), then flip to p(me_i wins)
    if the mover isn't the search root's side. Never-throw: any failure (missing keys
    the model wasn't ready for, a NaN/out-of-range prediction) folds to 0.5
    (uninformative) rather than propagating -- a single bad leaf must not void an
    entire playout budget's worth of work in the caller."""
    try:
        mover = (raw_obs.get("current") or {}).get("yourIndex", me_i)
        feat = value_api["value_features"](raw_obs, mover)
        p = float(value_api["value_predict"](feat))
    except Exception:
        return 0.5
    if p != p:  # NaN
        return 0.5
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    return p if mover == me_i else 1.0 - p


# ── Engine-call helpers ──────────────────────────────────────────────────────

_STEP_ERRORS = {
    1: "no element with the specified search_id",
    2: "released item",
    3: "battle has ended",
    4: "minCount <= len(select) <= maxCount violated",
    5: "select index out of range",
    6: "duplicate select elements",
    30: "agent_ptr broken",
}


def _step_raw(search_id: int, select: list, api) -> tuple:
    """[v2] Low-level SearchStep, bypassing json_to_dataclass -- ports the call signature
    and error-code table from the typed wrapper (cg/api.py:597-627) but returns the raw
    parsed dict straight off the engine's JSON (see module docstring's "KEY UNLOCK"
    section). Only ever called with a search_id that came from a typed `api['search_begin']`
    call or from a prior `_step_raw` call -- by that point `api['capi_mod'].agent_ptr` is
    guaranteed to exist (the typed search_begin sets it on first use, cg/api.py:541-544)."""
    lib = api["lib"]
    agent_ptr = api["capi_mod"].agent_ptr
    arr = (ctypes.c_int * len(select))(*select)
    bs = lib.SearchStep(agent_ptr, search_id, arr, len(select))
    result = json.loads(bs.decode())
    err = result.get("error", 0)
    if err != 0:
        raise RuntimeError(f"SearchStep error {err}: {_STEP_ERRORS.get(err, 'unknown')}")
    state = result["state"]
    return int(state["searchId"]), state["observation"]


def _safe_release(sid, api) -> None:
    try:
        api["search_release"](sid)
    except Exception:
        pass


def _playout(root_sid: int, idx: int, me_i: int, matchup: str, deadline: float, api,
             bc_api, value_api, step_budget: list):
    """[v2] One Monte-Carlo playout for BC candidate `idx`, branched off `root_sid`. Own
    implementation of the scheme in submissions/scout_deeper/main.py:426-544's playout():
    play forward counting turn-owner handovers, stop at DEPTH handovers/terminal/step-cap/
    deadline and evaluate the leaf. Divergence from that reference: scout dispatches BOTH
    sides through one shared full-info heuristic every step; this function dispatches
    per-actor via `_pick_for_actor` (BC for our own side always, BC for the opponent only
    in the mirror matchup, generic scorer otherwise) -- a single parameterized loop
    handles both "my remaining turn" and "their turn" instead of two near-duplicate
    phases (that was v1's structure; v2 collapses it further since the leaf is no longer
    computed differently depending on which phase it was reached from). Returns a float in
    [0,1], or None if the very first step (root_sid -> idx) itself failed -- callers must
    skip None rather than counting it as a misleading 0.0 (a real loss)."""
    try:
        if step_budget[0] <= 0:
            return None
        step_budget[0] -= 1
        sid, raw_obs = _step_raw(root_sid, [idx], api)
    except Exception:
        return None

    owner = (raw_obs.get("current") or {}).get("yourIndex", me_i)
    handovers = 0
    steps = 0
    while True:
        cur = raw_obs.get("current") or {}
        result = cur.get("result")
        if result is not None and result >= 0:
            val = _terminal_value(result, me_i)
            _safe_release(sid, api)
            return val
        sel = raw_obs.get("select")
        if sel is None:
            val = _value_leaf(raw_obs, me_i, value_api)
            _safe_release(sid, api)
            return val
        now = cur.get("yourIndex", owner)
        if now != owner:
            owner = now
            handovers += 1
            if handovers >= DEPTH:
                val = _value_leaf(raw_obs, me_i, value_api)
                _safe_release(sid, api)
                return val
        if steps >= MAX_PLAYOUT_STEPS or time.monotonic() > deadline or step_budget[0] <= 0:
            val = _value_leaf(raw_obs, me_i, value_api)
            _safe_release(sid, api)
            return val
        choice = _pick_for_actor(raw_obs, now, me_i, matchup, bc_api, api)
        if not choice:
            val = _value_leaf(raw_obs, me_i, value_api)
            _safe_release(sid, api)
            return val
        try:
            step_budget[0] -= 1
            sid2, raw_obs2 = _step_raw(sid, choice, api)
        except Exception:
            val = _value_leaf(raw_obs, me_i, value_api)
            _safe_release(sid, api)
            return val
        _safe_release(sid, api)  # linear walk, no backtracking -- parent never needed again
        sid, raw_obs = sid2, raw_obs2
        steps += 1


def _ucb1_pick(cand: list, acc: dict, n_eval: dict) -> int:
    """[v2] UCB1 arm selection: every candidate gets one sample first (round-robin over
    `cand` in BC-score order, since `cand` is already BC-sorted with the incumbent
    prepended -- see decide()), then argmax(mean + UCB_C*sqrt(ln(N)/n)). Own small pure
    function rather than scout's inline UCB branch (submissions/scout_deeper/main.py:
    525-532), so it's unit-testable independent of the engine. Tie-break: smaller index
    (this project's standing convention, e.g. v1's `(-score, i)` sorts)."""
    untried = [i for i in cand if n_eval[i] == 0]
    if untried:
        return untried[0]
    total_n = sum(n_eval[i] for i in cand)
    log_n = math.log(max(total_n, 1))
    return max(cand, key=lambda i: (
        acc[i] / n_eval[i] + UCB_C * math.sqrt(log_n / n_eval[i]), -i
    ))


def _new_stats() -> dict:
    return {
        "ok": False, "reason": "init", "ms": 0.0, "spent": _SPENT["t"],
        "ndet": 0, "n_cands": 0, "margin": 0.0, "override": False,
        "p_best": 0.0, "p_inc": 0.0,
    }


def decide(obs_dict: dict, bc_scores, bc_order, bc_pick, matchup, my_deck):
    """Never-throw. Returns (override_pick | None, stats).

    `bc_scores`/`bc_order`: per-option BC scores and the (-score, idx) sorted index order
    for the CURRENT decision's options, from the same obs_dict. `bc_pick`: the pick
    _bc_decide already returned for this decision (the incumbent the search may
    override). `matchup`: label from main.py's _detect_matchup_raw (or
    "generic"/"mirror"). `my_deck`: our own 60-card deck.csv contents (module-level
    `my_deck` in main.py).

    [v1 fixer p.1] `_SPENT["n_exceptions"]` is cumulative across the whole episode: it
    counts BOTH exceptions that abort a whole decide() call (outer except below) AND
    exceptions that void a single playout inside the UCB1 loop (folded into `det_exceptions`
    then added to `_SPENT["n_exceptions"]` at the end) -- a persistent per-playout failure
    mode still reaches MAX_SEARCH_EXCEPTIONS and trips the sticky disable, but an isolated
    transient failure doesn't accumulate forever (the counter resets to 0 on any call that
    completes with zero playout-level exceptions).
    """
    stats = _new_stats()
    if _SPENT["disabled"]:
        stats["reason"] = "hard_stop"
        return None, stats

    t0 = time.monotonic()
    try:
        sel = obs_dict.get("select") or {}
        options = sel.get("option") or []
        n_options = len(options)
        if n_options == 0 or len(bc_scores) != n_options or len(bc_order) != n_options:
            stats["reason"] = "bc_mismatch"
            return None, stats
        if len(bc_pick) != 1:
            stats["reason"] = "multi_pick"
            return None, stats
        ctx = sel.get("context")
        cur_raw = obs_dict.get("current") or {}
        turn = int(cur_raw.get("turn") or 0)

        api = _get_api()
        if api is None:
            stats["reason"] = "no_api"
            return None, stats
        value_api = _get_value_api()
        if value_api is None:
            # [v2] Parallel track not landed yet -- search stays fully unavailable, no
            # fallback to the refuted v1 linear leaf. See module docstring.
            stats["reason"] = "no_value_model"
            return None, stats
        if ctx is None or int(ctx) != int(api["SelectContext"].MAIN):
            stats["reason"] = "not_main"
            return None, stats
        if not (MIN_OPTIONS <= n_options <= MAX_OPTIONS):
            stats["reason"] = "n_options"
            return None, stats
        if turn < MIN_TURN:
            stats["reason"] = "turn"
            return None, stats
        if not obs_dict.get("search_begin_input"):
            stats["reason"] = "no_sbi"
            return None, stats

        n_det = N_DET
        if _SPENT["t"] > EPISODE_HARD_S:
            _SPENT["disabled"] = True
            stats["reason"] = "hard_stop"
            return None, stats
        if _SPENT["t"] > EPISODE_SOFT_S:
            n_det = 1
            _SPENT["n_degraded"] += 1

        incumbent = bc_pick[0]
        cand = [i for i in bc_order[:K_CAND] if 0 <= i < n_options]
        if incumbent not in cand:
            cand = [incumbent] + cand
        stats["n_cands"] = len(cand)
        if len(cand) < 2:
            stats["reason"] = "single_cand"
            return None, stats

        try:
            obs_root = api["to_observation_class"](obs_dict)
        except Exception:
            stats["reason"] = "to_obs_failed"
            return None, stats
        st = obs_root.current
        if st is None or obs_root.select is None:
            stats["reason"] = "no_state"
            return None, stats
        me_i = st.yourIndex

        opp_template = my_deck if matchup == "mirror" else _TEMPLATES.get(matchup)
        bc_api = _get_bc_api()
        bc_mod = bc_api["module"] if bc_api is not None else None

        deadline = t0 + TIME_BUDGET_S
        step_budget = [MAX_ENGINE_STEPS_PER_DECISION]
        acc = {i: 0.0 for i in cand}
        n_eval = {i: 0 for i in cand}
        worlds: list = []
        det_exceptions = 0
        search_begin_f = api["search_begin"]
        search_end_f = api["search_end"]

        # [v2] Snapshot bc_features_2b's turn-history globals BEFORE any rollout call can
        # touch them, restore in `finally` unconditionally -- see module docstring's "RISK
        # FOUND AND HANDLED" section. Must wrap the WHOLE playout loop (any exception
        # inside it still needs the restore to run).
        hist_snapshot = None
        if bc_mod is not None:
            hist_snapshot = (bc_mod._turn_hist_turn, list(bc_mod._turn_hist_buf),
                              bc_mod._turn_hist_last_sig, bc_mod._turn_hist_ability_used)

        try:
            for t in range(N_PLAYOUTS):
                if time.monotonic() > deadline or step_budget[0] <= 0:
                    break
                pick_i = _ucb1_pick(cand, acc, n_eval)
                slot = t % n_det if n_det > 0 else 0
                if slot < len(worlds):
                    root_sid = worlds[slot]
                elif len(worlds) < n_det:
                    try:
                        hidden = _sample_hidden(st, me_i, my_deck, opp_template)
                        root = search_begin_f(obs_root, **hidden)
                        worlds.append(root.searchId)
                        root_sid = worlds[-1]
                    except Exception:
                        det_exceptions += 1
                        root_sid = worlds[slot % len(worlds)] if worlds else None
                else:
                    root_sid = worlds[slot % len(worlds)] if worlds else None
                if root_sid is None:
                    break  # no world could ever be created -- nothing left to sample
                try:
                    val = _playout(root_sid, pick_i, me_i, matchup, deadline, api,
                                    bc_api, value_api, step_budget)
                except Exception:
                    det_exceptions += 1
                    val = None
                if val is not None:
                    acc[pick_i] += val
                    n_eval[pick_i] += 1
        finally:
            for sid in worlds:
                _safe_release(sid, api)
            try:
                search_end_f()
            except Exception:
                pass
            if hist_snapshot is not None:
                (bc_mod._turn_hist_turn, bc_mod._turn_hist_buf,
                 bc_mod._turn_hist_last_sig, bc_mod._turn_hist_ability_used) = hist_snapshot

        elapsed = time.monotonic() - t0
        _SPENT["t"] += elapsed
        _SPENT["n_calls"] += 1
        stats["ms"] = elapsed * 1000.0
        stats["spent"] = _SPENT["t"]
        stats["ndet"] = len(worlds)

        if det_exceptions > 0:
            _SPENT["n_exceptions"] += det_exceptions
            if _SPENT["n_exceptions"] >= MAX_SEARCH_EXCEPTIONS:
                _SPENT["disabled"] = True
                stats["ok"] = True
                stats["reason"] = "hard_stop_det_exceptions"
                return None, stats
        else:
            _SPENT["n_exceptions"] = 0

        n_top = n_eval.get(incumbent, 0)
        if n_top == 0:
            stats["ok"] = True
            stats["reason"] = "no_eval"
            return None, stats
        evaluated = [i for i in cand if n_eval[i] == n_top]
        avg = {i: acc[i] / n_eval[i] for i in evaluated}
        best = max(evaluated, key=lambda i: (avg[i], bc_scores[i], -i))
        stats["ok"] = True
        stats["p_best"] = avg[best]
        stats["p_inc"] = avg.get(incumbent, 0.0)
        if best == incumbent:
            stats["reason"] = "no_gain"
            return None, stats
        margin = avg[best] - avg[incumbent]
        stats["margin"] = margin
        if margin < DELTA_P:
            stats["reason"] = "margin"
            return None, stats
        stats["override"] = True
        stats["reason"] = "override"
        _SPENT["n_overrides"] += 1
        return [best], stats

    except Exception as exc:
        _SPENT["n_exceptions"] += 1
        if _SPENT["n_exceptions"] >= MAX_SEARCH_EXCEPTIONS:
            _SPENT["disabled"] = True
        elapsed = time.monotonic() - t0
        _SPENT["t"] += elapsed
        stats["ms"] = elapsed * 1000.0
        stats["spent"] = _SPENT["t"]
        stats["reason"] = f"exception:{exc!r}"[:160]
        return None, stats
