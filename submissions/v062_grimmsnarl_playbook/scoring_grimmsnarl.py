"""v061 grimmsnarl_heuristic — deterministic scoring module (Phase 1+2).

v036-style greedy heuristic scorer (score_option -> choose_options), adapted to the
jiatu.l Grimmsnarl ex / Munkidori toolbox deck. See DESIGN.md for the full spec this
module implements. This file is PURE scoring: it operates only on the typed obs
(to_observation_class output) and never touches the raw obs_dict. Guards (promotion,
lethal-KO, energy-attach, Adrena-Brain trigger) live in main.py and run BEFORE this
module is ever reached; this module is only responsible for the "no guard fired"
remainder, exactly like v036_archaludon_patched/main.py's score_option/choose_options.

Deck concept (see DESIGN.md for full detail):
  Impidimp (646) -> Morgrem (647) -> Grimmsnarl ex (648), the only real attackers
  (Filch/Corkscrew Punch/Shadow Bullet). Munkidori (112, Adrena-Brain) and Froslass
  (104, Freezing Shroud, evolves from Snorunt 860) are an anti-tank damage-counter
  engine: Freezing Shroud auto-feeds a counter onto every Ability-holder each Checkup,
  Adrena-Brain moves up to 3 of those counters onto the opponent's Pokemon once/turn.
  Grimmsnarl ex's Punk Up (fires on evolve-from-hand) searches up to 5 Basic Darkness
  Energy from the deck and free-attaches them to Marnie's Pokemon -- the deck's single
  biggest tempo spike (0 -> loaded attacker in one turn).

Scoring convention (mirrors v036): board development (bench/evolve/attach/trainers)
scores ~1000-28000 ("do first"); attacking/using an ability that ends the turn scores
in the raw-damage range (~-100..200) so choose_options naturally develops the board
before it commits to ending the turn.

v062 [форензик] edits (each individually tagged "[форензик v062]" at its own definition):
  1. reset_game_state() bug fix -- was called by main.py but never defined, so all
     module-level state (opp-attack tracker, and now the item-2 sticky matchup) leaked
     across games sharing one Kaggle process.
  2. detect_matchup/opp_has_tank were dead code (defined, never called) -- rebuilt as a
     sticky board-scan (crustle>archaludon>lucario>tr_mewtwo>abomasnow>mirror>alakazam>
     generic, ported from v049_bench_sticky_guard's board+sticky layer) and wired into
     score_option (per-decision refresh), score_boss/score_target (item 5), and the
     [dec] trace log (m={matchup} tag).
  4. score_retreat: retreating an active, uncastable, Dark-weak Munkidori for a real
     bench attacker is a strict improvement, not a cost -- was a flat -100 always.
  5. score_boss/score_target: vs tr_mewtwo, prioritize gusting Team Rocket's Spidops
     (401, their real attacker) over Mewtwo ex (mostly inert behind Power Saver).
  6. score_play (Spikemuth branch): replacing a live opponent stadium is worth more than
     a routine first play (Lucario/Gravity Mountain prep).
Item 3 (Munkidori energy guard) lives in main.py, not here -- see that file's docstring.

v062 [майнинг v062] wave-2 edits on top of the forensic pass above, from mining 2711 pure
jiatu.l/bono expert games (docs/gold/v062_expert_playbook_lines.md; full aggregates in
data/il_grimmsnarl_v2/expert_playbook.json). Each is tagged "[майнинг v062]" at its own
definition:
  3. Boss's Orders (BOSS_TARGET_PRIORITY, _boss_tier_bonus, wired into _boss_target_key /
     score_boss / score_target's SWITCH-TO_ACTIVE branch): REPLACES the wave-1
     tr_mewtwo/Spidops-only flat bonus with matchup-specific priority tables mined per-
     matchup -- crustle: Mega Kangaskhan ex(756) > Dwebble(344) > Crustle(345) itself
     (playbook lines 39-43, the wall is comparatively DE-prioritized in wins); alakazam:
     Fezandipiti ex(140) > Abra/Kadabra/Alakazam line (lines 62-66); tr_mewtwo: TR Mewtwo
     ex(431) itself FIRST, Spidops(401) second (lines 97-102 -- flips wave-1's Spidops-first
     table: sniping the ex while it's still locked behind Power Saver denies it ever
     attacking, and is 63% of Boss uses in wins vs losses wasting Boss on secondary support);
     archaludon: Duraludon(169) (line 112). mirror: no table entry by design (see
     BOSS_TARGET_PRIORITY's own docstring -- Munkidori is already covered by the killable
     branch, and over-fixating on it correlates with LOSSES in the data). Also adds a small
     universal "prefer an already-damaged target" bonus (line 68-70: median HP% pulled is
     81% in wins vs 100% in losses).
  1/2/4 (Munkidori-energy gate removal, Adrena-Brain bench-routing, retreat verification)
     live in main.py -- see that file's module docstring wave-2 section.
"""

import sys as _sys

from cg.api import (
    AreaType,
    LogType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

try:
    from cg.api import all_attack
    ALL_ATTACKS = {a.attackId: a for a in all_attack()}
except Exception:
    ALL_ATTACKS = {}

# ── Card IDs ──────────────────────────────────────────────────────────────

GRIMMSNARL = 648   # Stage2 ex, HP320, weak Grass. Shadow Bullet {D}{D}=180 (+30 bench).
                   # Ability Punk Up: on evolve-from-hand, search up to 5 Basic {D}
                   # Energy and attach freely to Marnie's Pokemon. WIN-CON.
MORGREM = 647      # Stage1, HP100. Corkscrew Punch {D}{D}=60.
IMPIDIMP = 646     # Basic, HP70. Filch {0}=draw / Corkscrew Punch {D}=10.
MUNKIDORI = 112    # Basic, HP110, weak Dark. Mind Bend UNCASTABLE (0 {P} in deck).
                   # Ability Adrena-Brain: once/turn, with any {D} attached, move up
                   # to 3 damage counters from 1 of ours to 1 of opponent's.
FROSLASS = 104     # Stage1 (from Snorunt), HP90, weak Metal. Frost Smash UNCASTABLE
                   # (0 {W} in deck). Ability Freezing Shroud: each Checkup, +1 damage
                   # counter on every Ability-holder (both sides, except Froslass) --
                   # auto-feeds Adrena-Brain every turn. Pure passive engine piece.
SNORUNT = 860      # Basic, HP70. Chilly UNCASTABLE. Only value = evolve into Froslass.

MARNIE_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL}  # Punk Up targets / Spikemuth searches

# Only real attackers: 648 (180), 647 (60), 646 (10 or draw). 112/104/860 never attack.
_ATTACK_BASE_DMG = {
    937: 180,  # Shadow Bullet (648)
    936: 60,   # Corkscrew Punch (647, Morgrem)
    935: 10,   # Corkscrew Punch (646, Impidimp)
    934: 0,    # Filch (646, draw -- no damage)
}
SHADOW_BULLET = 937
CORKSCREW_MORGREM = 936
CORKSCREW_IMPIDIMP = 935
FILCH = 934

# Trainers / Stadium / Energy
POFFIN = 1086          # Buddy-Buddy Poffin (Item): search 2 Basic <=70HP to bench.
POKE_PAD = 1152        # Poke Pad (Item): search a non-Rule-Box Pokemon to hand.
LILLIE = 1227          # Lillie's Determination (Supporter): shuffle hand, draw 6(8).
PETREL = 1219          # Team Rocket's Petrel (Supporter): search a Trainer to hand.
RARE_CANDY = 1079      # Rare Candy (Item): Basic -> Stage2 skip.
NIGHT_STRETCHER = 1097 # Night Stretcher (Item): Pokemon or Basic Energy from discard.
UNFAIR_STAMP = 1080    # Unfair Stamp (Item, ACE SPEC): refill+disruption after a KO.
BOSS = 1182            # Boss's Orders (Supporter): gust 1 opp benched Pokemon.
DAWN = 1231            # Dawn (Supporter): search a Basic + Stage1 + Stage2 to hand.
XEROSIC = 1197         # Xerosic's Machinations (Supporter): opp discards down to 3.
POKEGEAR = 1122        # Pokegear 3.0 (Item): look top 7, take a Supporter.
SPIKEMUTH = 1259       # Spikemuth Gym (Stadium): once/turn, search a Marnie's Pokemon.
D_ENERGY = 7           # Basic Darkness Energy (also EnergyType.DARKNESS's value == 7).

ITEMS = {POFFIN, POKE_PAD, RARE_CANDY, NIGHT_STRETCHER, UNFAIR_STAMP, POKEGEAR}
SUPPORTERS = {LILLIE, PETREL, BOSS, DAWN, XEROSIC}

_SETUP_ACTIVE_PRIORITY = {
    IMPIDIMP: (100000, "active Impidimp line"),
    SNORUNT: (3000, "active Snorunt"),
    MUNKIDORI: (1000, "active Munkidori fallback"),
}

# Matchup detection (see DESIGN.md "Matchup detection")
CRUSTLE_LINE = {344, 345}          # Rock Inn wall -- Shadow Bullet blocked when 345 active
ARCHALUDON_LINE = {169, 190}       # tank beatdown
LUCARIO_LINE = {677, 678}          # [форензик v062] item 2 -- Lucario ex tempo deck
TR_MEWTWO_LINE = {431}             # [форензик v062] item 2 -- Team Rocket's Mewtwo ex control
ABOMASNOW_LINE = {722, 723}        # [форензик v062] item 2 -- Abomasnow tank line
MIRROR_LINE = {646, 647, 648}      # [форензик v062] item 2 -- opponent's board shows our OWN
                                    # Marnie Impidimp/Morgrem/Grimmsnarl-ex IDs -> mirror match
ALAKAZAM_LINE = {741, 742, 743}    # our best matchup -- Powerful Hand OHKO deck
SPIDOPS = 401                      # [форензик v062] item 5 -- Team Rocket's Spidops, TR Mewtwo's
                                    # real attacker (Rocket Rush, scales w/ TR-Pokemon on board)

# [майнинг v062] item 3 -- Boss's Orders matchup-specific target-priority ids, mined from
# 2711 pure jiatu.l/bono expert games (docs/gold/v062_expert_playbook_lines.md).
MEGA_KANGASKHAN_EX = 756  # crustle: single most-pulled Boss target in the WHOLE dataset (line 39-43).
DWEBBLE = 344              # crustle: Crustle's own pre-evo (== CRUSTLE_LINE's 344).
CRUSTLE = 345              # crustle: the wall itself -- de-prioritized vs the two ids above it.
FEZANDIPITI_EX = 140       # alakazam: 70% of all Boss uses in wins (line 62-66).
TR_MEWTWO_EX = 431         # tr_mewtwo: the ex itself, locked behind Power Saver until 4+ TR mons
                           # in play -- 63% of Boss uses in wins target it directly (line 97-102).
DURALUDON = 169            # archaludon: pre-evo, "deny the evolve" (line 112).

# [майнинг v062] item 3 -- ordered priority tiers per matchup (highest-priority tier first);
# a target's tier bonus is (len(tiers) - rank) * _BOSS_TIER_WEIGHT via _boss_tier_bonus below.
# Replaces the wave-1 hardcoded tr_mewtwo/Spidops-only bonus in _boss_target_key/score_target.
# "mirror" intentionally has NO entry: playbook lines 76-79/166 show Munkidori (112) is
# 55%/84% of Boss uses in mirror wins/losses, but (a) the killable-branch of score_target
# already covers it (Munkidori is squishy, usually one-shot-able) and (b) the 84%-in-LOSSES
# split means over-fixating on it correlates with losing, not winning -- adding a redundant
# bonus on top of the killable branch risks the same over-focus the losing games show.
BOSS_TARGET_PRIORITY = {
    "crustle": (frozenset({MEGA_KANGASKHAN_EX}), frozenset({DWEBBLE}), frozenset({CRUSTLE})),
    "alakazam": (frozenset({FEZANDIPITI_EX}), frozenset(ALAKAZAM_LINE)),
    "tr_mewtwo": (frozenset({TR_MEWTWO_EX}), frozenset({SPIDOPS})),
    "archaludon": (frozenset({DURALUDON}),),
}
_BOSS_TIER_WEIGHT = 6000  # roughly matches the wave-1 flat Spidops bonus magnitude (5000-8000)
                           # so a table hit still dominates the prize/energy tie-break terms.


def _boss_tier_bonus(matchup, card_id):
    """[майнинг v062] item 3 -- (len(tiers)-rank)*_BOSS_TIER_WEIGHT if card_id is in
    BOSS_TARGET_PRIORITY[matchup]'s tiers (earlier tier = bigger bonus), else 0 (falls back
    to the plain prize/energy heuristic -- e.g. every "mirror" target, or any matchup/id not
    in the table)."""
    tiers = BOSS_TARGET_PRIORITY.get(matchup, ())
    for rank, tier in enumerate(tiers):
        if card_id in tier:
            return (len(tiers) - rank) * _BOSS_TIER_WEIGHT
    return 0


CARD_DB = {c.cardId: c for c in all_card_data()}

# Track opponent's last-turn attack via logs (ported from v036; Boss logic in this
# deck doesn't currently key off it, but it's harmless to keep populated in case a
# future edit or the legibility harness wants it).
_opp_last_attack_id = None
_cur_turn_logs = []

# [форензик v062] item 2 -- sticky matchup label (see detect_matchup below). Module-level
# so it survives across the whole game; must be reset on every new-game boundary via
# reset_game_state() (item 1) since a Kaggle process can play multiple games back to back.
_last_matchup = None


def _update_opp_attack_tracking(obs):
    global _opp_last_attack_id, _cur_turn_logs
    yi = obs.current.yourIndex
    for entry in obs.logs:
        if entry.type == LogType.TURN_END:
            for prev in _cur_turn_logs:
                if prev.type == LogType.ATTACK and getattr(prev, 'playerIndex', yi) != yi:
                    _opp_last_attack_id = prev.attackId
            _cur_turn_logs.clear()
        else:
            _cur_turn_logs.append(entry)


def reset_game_state():
    """[форензик v062] item 1 (bug fix, done first). main.py's agent() has always called
    getattr(_scorer, 'reset_game_state', None) on every new-game boundary (obs.select is
    None) -- but v061 never DEFINED this function, so the call was a silent no-op
    (getattr returns None, main.py's `if _reset is not None` guard skips it) and every
    module-level global below leaked across games within the same long-lived Kaggle
    process (a single process can play several episodes back to back). Resets ALL
    scorer-side sticky state added so far: the opponent-last-attack tracker (pre-existing,
    v061) and the new sticky matchup label (item 2, this file)."""
    global _opp_last_attack_id, _cur_turn_logs, _last_matchup
    _opp_last_attack_id = None
    _cur_turn_logs = []
    _last_matchup = None


# ── Board helpers (ported from v036, adapted where noted) ──────────────────

def read_deck_csv():
    import os
    fp = "deck.csv"
    if not os.path.exists(fp):
        fp = "/kaggle_simulations/agent/deck.csv"
    with open(fp) as f:
        return [int(line) for line in f.read().strip().split("\n")]


def get_card(obs, area, index, player_index):
    if area is None or index is None:
        return None
    ps = obs.current.players[player_index]
    if area == AreaType.DECK and obs.select and obs.select.deck is not None:
        return obs.select.deck[index] if index < len(obs.select.deck) else None
    if area == AreaType.HAND and ps.hand is not None:
        return ps.hand[index] if index < len(ps.hand) else None
    if area == AreaType.DISCARD:
        return ps.discard[index] if index < len(ps.discard) else None
    if area == AreaType.ACTIVE:
        return ps.active[index] if index < len(ps.active) else None
    if area == AreaType.BENCH:
        return ps.bench[index] if index < len(ps.bench) else None
    if area == AreaType.PRIZE:
        return ps.prize[index] if index < len(ps.prize) else None
    if area == AreaType.STADIUM:
        return obs.current.stadium[index] if index < len(obs.current.stadium) else None
    if area == AreaType.LOOKING and obs.current.looking is not None:
        return obs.current.looking[index] if index < len(obs.current.looking) else None
    return None


def option_card(obs, opt):
    yi = obs.current.yourIndex
    pi = opt.playerIndex if opt.playerIndex is not None else yi
    if opt.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, opt.index, pi)
    return get_card(obs, opt.area, opt.index, pi)


def option_target(obs, opt):
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    return get_card(obs, opt.inPlayArea, opt.inPlayIndex, obs.current.yourIndex)


def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def active_pokemon(obs):
    ps = my_state(obs)
    return ps.active[0] if ps.active else None


def opp_active_pokemon(obs):
    ps = opp_state(obs)
    return ps.active[0] if ps.active else None


def opp_bench_pokemon(obs):
    return [p for p in opp_state(obs).bench if p]


def all_my_pokemon(obs):
    ps = my_state(obs)
    return [p for p in (ps.active + ps.bench) if p]


def hand_ids(obs):
    hand = my_state(obs).hand
    return [c.id for c in hand if c] if hand else []


def discard_ids(obs):
    return [c.id for c in (my_state(obs).discard or []) if c]


def energy_count(pokemon):
    """Total attached energy (any type)."""
    if pokemon is None:
        return 0
    if getattr(pokemon, "energyCards", None) is not None:
        return len(pokemon.energyCards)
    return len(getattr(pokemon, "energies", []) or [])


def d_energy_count(pokemon):
    """Count of Basic Darkness Energy (card id 7) attached -- our only energy type."""
    if pokemon is None:
        return 0
    cards = getattr(pokemon, "energyCards", None)
    if cards is not None:
        return sum(1 for c in cards if c and c.id == D_ENERGY)
    return 0


def retreat_cost(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    return getattr(data, "retreatCost", 0) if data else 0


def damage_on(pokemon):
    if pokemon is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", pokemon.hp) - pokemon.hp)


def has_tool(pokemon):
    return bool(getattr(pokemon, "tools", []) or [])


def count_in_play(obs, card_id):
    return sum(1 for p in all_my_pokemon(obs) if p.id == card_id)


def has_in_play(obs, card_id):
    return any(p.id == card_id for p in all_my_pokemon(obs))


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def is_dark_weak(pokemon):
    if pokemon is None:
        return False
    data = CARD_DB.get(pokemon.id)
    w = getattr(data, "weakness", None) if data else None
    if w is None:
        return False
    return getattr(w, "value", w) == D_ENERGY  # EnergyType.DARKNESS == 7 == D_ENERGY


def effective_damage(base_damage, target):
    """base*2 if target is weak to Dark; our only damage source (Shadow Bullet /
    Corkscrew Punch) is Dark-type."""
    return base_damage * 2 if is_dark_weak(target) else base_damage


def _first_option_index(obs, card_id):
    for o in obs.select.option:
        oc = option_card(obs, o)
        if oc and oc.id == card_id:
            return getattr(o, 'index', None)
    return None


def _is_pokemon_card(card):
    """Duck-type: a Pokemon-in-play object has an 'hp' attribute; a plain Card
    (hand/discard energy or trainer) does not."""
    return card is not None and hasattr(card, "hp")


# ── Deck-line helpers ────────────────────────────────────────────────────

def can_evolve_to_grimmsnarl_now(obs):
    """True iff we can Rare Candy an Impidimp straight into Grimmsnarl ex right now:
    648 in hand, an Impidimp in play that was NOT put down this turn, turn >= 1."""
    if GRIMMSNARL not in hand_ids(obs):
        return False
    if getattr(obs.current, "turn", 0) < 1:
        return False
    for p in all_my_pokemon(obs):
        if p.id == IMPIDIMP and not getattr(p, "appearThisTurn", True):
            return True
    return False


def boss_attacker_damage(obs):
    """Base damage (pre-weakness) our current Active could deal if it attacked right
    now: 180 for a loaded Grimmsnarl ex, 60 for a loaded Morgrem, else 0."""
    active = active_pokemon(obs)
    if active is None:
        return 0
    if active.id == GRIMMSNARL and d_energy_count(active) >= 2:
        return 180
    if active.id == MORGREM and d_energy_count(active) >= 2:
        return 60
    return 0


def lethal_gust_exists(obs):
    """True iff there's a benched opponent Pokemon our current attacker could KO if
    Boss's Orders gusted it to Active."""
    dmg = boss_attacker_damage(obs)
    if dmg <= 0:
        return False
    return any(effective_damage(dmg, t) >= t.hp for t in opp_bench_pokemon(obs))


def bench_thin(obs):
    return len([p for p in my_state(obs).bench if p]) <= 1


def key_mon_in_discard(obs):
    disc = discard_ids(obs)
    return any(c in (GRIMMSNARL, MORGREM) for c in disc)


def energy_starved(obs):
    board_has_d = any(d_energy_count(p) > 0 for p in all_my_pokemon(obs))
    hand_has_d = D_ENERGY in hand_ids(obs)
    return not board_has_d and not hand_has_d


def poke_pad_line_incomplete(obs):
    """True while we still need one of Impidimp/Morgrem/Munkidori and none is in
    hand (648 itself is Rule Box -- Poke Pad can never fetch it)."""
    ids_hand = hand_ids(obs)
    lacking = not (has_in_play(obs, IMPIDIMP) or has_in_play(obs, MORGREM)) or not has_in_play(obs, MUNKIDORI)
    none_in_hand = not any(c in ids_hand for c in (IMPIDIMP, MORGREM, MUNKIDORI))
    return lacking and none_in_hand


def attacker_line_started(obs):
    """True once at least one Marnie-line attacker (Impidimp/Morgrem/Grimmsnarl) is
    already in play -- i.e. the critical *starting* attacker piece is secured, and
    any further Impidimp/Morgrem pull from a search effect is an 'excess' 2nd+ copy,
    not the make-or-break first one.

    [v062 wave3 fix] Split out of the old shared `poke_pad_line_incomplete()` flag,
    which conflated "attacker line missing" with "Munkidori missing" -- that flag
    stayed True (triggering a 20000-priority re-fetch of Impidimp) purely because
    Munkidori was absent, even with an Impidimp already on board. Confirmed in trace
    `v009a_crustle_001.json` (out/v062_local_loss_forensic.md question 1): Poke Pad
    grabbed a redundant 2nd Impidimp on turns 1 and 3 while the only actually missing
    piece (Munkidori) sat unfetched until turn 15. This helper answers ONLY "is the
    attacker line itself started", decoupled from Munkidori's state."""
    return has_in_play(obs, IMPIDIMP) or has_in_play(obs, MORGREM) or has_in_play(obs, GRIMMSNARL)


def poffin_would_help(obs):
    """False once the bench is full, or once we already have both 646 and 860
    developed with a wide bench (nothing useful left for Poffin to fetch)."""
    bench_n = len([p for p in my_state(obs).bench if p])
    if bench_n >= 5:
        return False
    already_developed = has_in_play(obs, IMPIDIMP) and has_in_play(obs, SNORUNT) and bench_n >= 3
    return not already_developed


def night_stretcher_has_urgent_target(obs):
    disc = discard_ids(obs)
    if any(c in (GRIMMSNARL, MORGREM) for c in disc):
        return True
    if D_ENERGY in disc and energy_starved(obs):
        return True
    return False


def petrel_has_need(obs):
    ids = hand_ids(obs)
    if lethal_gust_exists(obs) and BOSS not in ids:
        return True
    if can_evolve_to_grimmsnarl_now(obs) and RARE_CANDY not in ids:
        return True
    if bench_thin(obs) and POFFIN not in ids:
        return True
    return False


# ── Matchup detection ────────────────────────────────────────────────────

def _detect_matchup_board_only(obs):
    """[форензик v062] item 2 -- pure board scan (active + bench, right now), priority-
    ordered: crustle > archaludon > lucario > tr_mewtwo > abomasnow > mirror > alakazam >
    generic. Ported/extended from v049_bench_sticky_guard's _detect_matchup_board_only
    (main.py:797-825 there) -- board layer only, no belief-net/torch dependency."""
    opp = opp_state(obs)
    ids = {p.id for p in (opp.active + opp.bench) if p}
    if ids & CRUSTLE_LINE:
        return "crustle"
    if ids & ARCHALUDON_LINE:
        return "archaludon"
    if ids & LUCARIO_LINE:
        return "lucario"
    if ids & TR_MEWTWO_LINE:
        return "tr_mewtwo"
    if ids & ABOMASNOW_LINE:
        return "abomasnow"
    if ids & MIRROR_LINE:
        return "mirror"
    if ids & ALAKAZAM_LINE:
        return "alakazam"
    return "generic"


def detect_matchup(obs):
    """[форензик v062] item 2 -- sticky wrapper around _detect_matchup_board_only, ported
    from v049_bench_sticky_guard's sticky pattern (main.py:827-848 there): board evidence
    wins immediately and unconditionally whenever a LINE id is currently visible (updates
    `_last_matchup`); once the board goes quiet ("generic" -- e.g. the identifying Pokemon
    got KO'd/benched-out-of-view), KEEP the last confidently-seen label instead of
    reverting to generic, since the opponent's deck archetype doesn't actually change
    mid-game. `_last_matchup` is reset per-game by reset_game_state() (item 1). Was dead
    code in v061 (detect_matchup/opp_has_tank were defined but never called from
    score_option/apply_overrides/score_boss/etc.) -- now wired into score_option (sticky
    refresh + [dec]-log tag), score_boss and score_target (item 5, now table-driven by
    [майнинг v062] item 3's BOSS_TARGET_PRIORITY)."""
    global _last_matchup
    label = _detect_matchup_board_only(obs)
    if label != "generic":
        _last_matchup = label
        return label
    if _last_matchup is not None:
        return _last_matchup
    return "generic"


def opp_has_tank(obs):
    """[форензик v062] now sticky (via detect_matchup) instead of a raw board re-scan --
    kept for API parity, still unused by any score_* branch in this file (main.py's own
    raw-dict _opp_has_tank / _detect_matchup_raw are what the guards actually read)."""
    return detect_matchup(obs) in ("crustle", "archaludon")


# ── Overrides (applied to every score_option result) ────────────────────

def apply_overrides(obs, opt, score, reason):
    # Hard rule: don't Lillie (draw 6/8, i.e. reduce deck by up to 8) with a low deck,
    # mirroring v036's Explorer guard. Exception ported from v036's Crustle branch:
    # a near-empty deck + a big hand is itself a deckout risk, so refilling via
    # shuffle-hand-into-deck-and-draw is *safe* in that specific corner.
    if opt.type == OptionType.PLAY:
        card = option_card(obs, opt)
        cid = card.id if card else None
        if cid == LILLIE:
            dc = my_state(obs).deckCount
            if dc <= 8:
                hc = my_state(obs).handCount
                if dc <= 3 and hc >= dc + 6:
                    return 15000, "low deck: Lillie refill exception"
                return -5000, "hard: don't draw with low deck"

    # Rock-Inn: Shadow Bullet does 0 real damage to an active Crustle 345 (Rock Inn
    # blocks ex-attacks) -- don't waste the turn, let Adrena-Brain grind / develop /
    # end instead. Gated tightly (exact attacker id + exact opponent Active id) so no
    # other matchup is touched.
    if opt.type == OptionType.ATTACK and getattr(opt, "attackId", None) == SHADOW_BULLET:
        active = active_pokemon(obs)
        opp_act = opp_active_pokemon(obs)
        if active and active.id == GRIMMSNARL and opp_act and opp_act.id == 345:
            return -5000, "Rock-Inn: Shadow Bullet does 0 vs active Crustle 345"

    return score, reason


# ── score_setup (SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON / MULLIGAN / IS_FIRST) ─

def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _SETUP_ACTIVE_PRIORITY.get(cid, (0, "unknown active"))
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return -10000, "never bench during setup"
    return 0, "non-setup"


# ── score_play (MAIN, PLAY) ───────────────────────────────────────────────

def _boss_target_key(obs, t, matchup):
    """[форензик v062] item 5 helper -- shared ranking key for score_boss's own best-target
    estimate and score_target's actual SWITCH/TO_ACTIVE target choice, so the two stay in
    sync. [майнинг v062] item 3 update: the wave-1 tr_mewtwo/Spidops-only flat bonus is
    replaced by the table-driven `_boss_tier_bonus` (BOSS_TARGET_PRIORITY, mined from 2711
    expert games -- see that table's own docstring), which now also covers crustle/
    alakazam/archaludon. Also folds in a small universal "prefer an already-damaged target"
    bonus (docs/gold/v062_expert_playbook_lines.md line 68-70: median HP% of the Boss target
    actually pulled is 81% in wins vs 100% in losses -- winning games finish already-hurt
    targets more often than they drag in a fresh full-HP one)."""
    tier_bonus = _boss_tier_bonus(matchup, t.id)
    damage_bonus = min(1000, damage_on(t) * 5)
    return tier_bonus + damage_bonus + prize_value(t) * 1000 + energy_count(t) * 100


def score_boss(obs):
    """Boss's Orders: LETHAL gust if it wins the game outright, else tempo-drag the
    scariest fresh threat on the opponent's bench (the plan's "83% of real usage"
    pattern -- not every Boss is a kill, most are tempo). [майнинг v062] item 3: weight the
    matchup-specific playbook target (BOSS_TARGET_PRIORITY -- Mega Kangaskhan ex vs crustle,
    Fezandipiti ex vs alakazam, TR Mewtwo ex vs tr_mewtwo, Duraludon vs archaludon) via
    _boss_target_key/_boss_tier_bonus."""
    dmg = boss_attacker_damage(obs)
    if dmg <= 0:
        return -500, "save Boss: no attacker"
    remaining = len(my_state(obs).prize)
    bench = opp_bench_pokemon(obs)
    matchup = detect_matchup(obs)
    for t in bench:
        if effective_damage(dmg, t) >= t.hp and prize_value(t) >= remaining:
            return 20000, "LETHAL Boss"
    if not bench:
        return -500, "save Boss: no bench target"
    best_t = max(bench, key=lambda t: _boss_target_key(obs, t, matchup))
    best = 2000 + _boss_target_key(obs, best_t, matchup)
    if _boss_tier_bonus(matchup, best_t.id) > 0:
        return best, f"Boss: tempo-drag playbook target ({matchup}: id={best_t.id})"
    return best, "Boss: tempo-drag scariest threat"


def _opp_stadium_harmful(obs):
    """[форензик v062] item 6 (stadium war, Lucario prep) -- True iff a stadium card the
    OPPONENT played is currently in play. Checked cg/api.py: State.stadium is a single
    GLOBAL slot (list[Card], size 0 or 1 -- only one stadium can be active at a time,
    playing a new one always replaces the old), and every Card carries its own
    `.playerIndex`, so ownership is directly observable from the typed obs -- no ID
    hardcoding or guessing needed. Playing our own Spikemuth Gym unconditionally replaces
    whatever stadium is in play, so ANY opponent stadium is worth clearing, not just one
    specific card; Gravity Mountain (id 1252, speeds up Lucario ex's retreat/attack tempo)
    is the concrete matchup this was written for, but the check is intentionally general."""
    try:
        stadium_cards = obs.current.stadium or []
        yi = obs.current.yourIndex
        return any(c and c.playerIndex != yi for c in stadium_cards)
    except Exception:
        return False


def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None

    # ── Basic Pokemon: bench them ──
    if cid == IMPIDIMP:
        return 18000, "bench Impidimp"
    if cid == MUNKIDORI:
        return 17500, "bench Munkidori (Adrena-Brain + Freezing Shroud synergy)"
    if cid == SNORUNT:
        return 14000, "bench Snorunt (enables Froslass)"

    # ── Stadium ──
    if cid == SPIKEMUTH:
        already_in_play = any(c and c.id == SPIKEMUTH for c in (obs.current.stadium or []))
        if already_in_play:
            return -200, "skip Spikemuth: already in play"
        if _opp_stadium_harmful(obs):
            # [форензик v062] item 6: replacing a live opponent stadium is worth more than
            # a routine first play (it also SEARCHES a Marnie's Pokemon -- pure upside).
            return 24000, "play Spikemuth Gym: replace opponent's stadium"
        return 20000, "play Spikemuth Gym"

    # ── Items: default 20000, only negative exceptions ──
    if cid in ITEMS:
        if cid == NIGHT_STRETCHER:
            if not night_stretcher_has_urgent_target(obs):
                return -500, "save Night Stretcher"
            return 20000, "play Night Stretcher"
        if cid == POFFIN:
            if not poffin_would_help(obs):
                return -500, "skip Poffin: bench full / no valid target"
            return 20000, "play Poffin"
        if cid == RARE_CANDY:
            if can_evolve_to_grimmsnarl_now(obs):
                return 21000, "play Rare Candy: Impidimp -> Grimmsnarl ex"
            return -500, "save Rare Candy: no ready evolve"
        if cid == POKE_PAD:
            if poke_pad_line_incomplete(obs):
                return 20000, "play Poke Pad: line incomplete"
            return 6000, "play Poke Pad: line ok"
        if cid == POKEGEAR:
            no_supp_in_hand = not any(c in SUPPORTERS for c in hand_ids(obs))
            if no_supp_in_hand and not obs.current.supporterPlayed:
                return 16000, "play Pokegear: need supporter"
            return 2000, "play Pokegear: low priority"
        if cid == UNFAIR_STAMP:
            return 19000, "play Unfair Stamp"
        return 20000, "play item"

    # ── Supporters: one per turn ──
    if cid in SUPPORTERS:
        if obs.current.supporterPlayed:
            return -1000, "supporter already played"
        if cid == BOSS:
            return score_boss(obs)
        if cid == XEROSIC:
            if opp_state(obs).handCount >= 4:
                return 6000, "play Xerosic: opp hand big"
            return -500, "save Xerosic: opp hand small"
        if cid == PETREL:
            if petrel_has_need(obs):
                return 6000, "play Petrel: concrete trainer need"
            return 2500, "play Petrel: generic search"
        if cid == LILLIE:
            ids = hand_ids(obs)
            if BOSS in ids and boss_attacker_damage(obs) > 0:
                return -500, "save Lillie: Boss in hand with attacker ready"
            if my_state(obs).handCount <= 2:
                return 9000, "play Lillie: low hand"
            return 5000, "play Lillie: refill"
        if cid == DAWN:
            if getattr(obs.current, "turn", 0) <= 4:
                return 8000, "play Dawn: early tutor"
            return -500, "save Dawn: late, dead tutor"

    return 1000, "generic play"


# ── score_evolve (MAIN EVOLVE, and ctx EVOLVE follow-up select) ─────────────

def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None

    if cid == GRIMMSNARL:
        # The payoff play: evolving into Grimmsnarl ex triggers Punk Up -> the whole
        # energy-attach engine turns on in one shot.
        target_is_active = opt.inPlayArea == AreaType.ACTIVE
        if target_is_active:
            bonus = 2000 if target and d_energy_count(target) >= 1 else 0
            return 28000 + bonus, "evolve into Grimmsnarl ex (Active, Punk Up)"
        return 20000, "evolve into Grimmsnarl ex (Bench)"
    if cid == MORGREM:
        target_is_active = opt.inPlayArea == AreaType.ACTIVE
        if target_is_active:
            return 16000, "evolve into Morgrem (Active)"
        return 12000, "evolve into Morgrem (Bench)"
    if cid == FROSLASS:
        return 14000, "evolve into Froslass (Freezing Shroud engine)"
    return 10000, "generic evolution"


# ── score_attach (MAIN ATTACH) + attach_target_score ─────────────────────

def attach_target_score(target, area):
    """Score attaching a Basic Darkness Energy to `target` (already in play).

    Priority ranks the ideal deficits highest (Grimmsnarl needs 2 for Shadow Bullet,
    Munkidori needs 1 to switch Adrena-Brain on), BUT every attach stays STRICTLY POSITIVE
    even for an already-"capped" target: on this deck energy on the board is always better
    than energy stranded in hand (it can't be Judged/discarded away and it advances the
    win-con), so the expert attaches its one energy EVERY turn rather than declining — never
    END-over-attach. Legibility probe 2026-07-21: over-cap targets used to score negative,
    so choose_options picked END and lost the free energy drop. The park-values below all beat
    END(0) yet sit far under board development (14000+), so a real setup play still wins."""
    if target is None:
        return 50
    cid = target.id
    e = d_energy_count(target)

    if cid == GRIMMSNARL:  # wants 2 for Shadow Bullet
        score = {0: 4000, 1: 7000}.get(e, 500)      # capped: still park a 3rd (safe), not negative
        if area == AreaType.ACTIVE:
            score += 1500
    elif cid == MUNKIDORI:  # needs 1 {D} to switch Adrena-Brain on; a 2nd is just parking
        score = 6000 if e == 0 else 400
    elif cid == MORGREM:  # wants 2 for Corkscrew {D}{D}
        score = {0: 3000, 1: 5000}.get(e, 450)
    elif cid == IMPIDIMP:  # 1 {D} enables its Corkscrew
        score = 2000 if e == 0 else 350
    else:
        score = 200  # 104/860/anything else never uses {D}, but a board attach still beats END

    # HP-ratio tweak (ported from v036): don't over-invest in a mon about to die.
    if getattr(target, "hp", 0) > 0:
        max_hp = getattr(target, "maxHp", target.hp)
        ratio = target.hp / max_hp if max_hp > 0 else 1
        if ratio <= 0.25:
            score -= 1500
        elif ratio <= 0.50:
            score -= 500
        else:
            score += min(1000, target.hp // 40 * 100)
    # Floor so any legal attach still beats declining (END=0) — spend the energy drop.
    return max(50, score)


def score_attach(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None

    if cid != D_ENERGY:
        return -500, "skip non-Dark energy"
    if obs.current.energyAttached:
        return -1000, "already attached"

    return attach_target_score(target, opt.inPlayArea), "attach Dark energy"


# ── ATTACK (MAIN ATTACK, and ctx ATTACK follow-up select) ────────────────

def attack_choice_score(obs, attack_id):
    """Weakness-adjusted damage for a given attackId. Falls back to raw damage if
    there's no opponent Active to weakness-check against."""
    base = _ATTACK_BASE_DMG.get(attack_id, 0)
    opp_act = opp_active_pokemon(obs)
    if opp_act is None:
        return base
    return effective_damage(base, opp_act)


def score_attack(obs, opt):
    attack_id = getattr(opt, "attackId", None)
    active = active_pokemon(obs)

    # Impidimp special-case: prefer Filch (draw) over Corkscrew Punch (10 dmg) unless
    # Corkscrew is lethal right now, or the hand is big enough that digging matters.
    if active and active.id == IMPIDIMP:
        if attack_id == CORKSCREW_IMPIDIMP:
            opp_act = opp_active_pokemon(obs)
            opp_hp = opp_act.hp if opp_act else 999
            if 10 >= opp_hp:
                return 200, "Impidimp: Corkscrew (lethal)"
            if my_state(obs).handCount >= 7:
                return 150, "Impidimp: Corkscrew (dig, big hand)"
            return -100, "Impidimp: save Corkscrew, prefer Filch"
        if attack_id == FILCH:
            return 100, "Impidimp: Filch (draw)"

    return attack_choice_score(obs, attack_id), "attack"


# ── RETREAT ────────────────────────────────────────────────────────────

def score_retreat(obs, opt):
    """[форензик v062] item 4 -- an active Munkidori (112) is dead weight, not a Pokemon
    worth guarding: its own attack (Mind Bend, {P} cost) is UNCASTABLE (0 {P} energy
    anywhere in this deck -- see module docstring), so it can never do anything by
    attacking, and Adrena-Brain works identically from the bench (the engine offers the
    OPT_ABILITY option regardless of area -- see main.py's _adrena_trigger_guard, which
    already scans active+bench for it). Worse, Munkidori is WEAK TO DARK and our own
    Shadow Bullet is a Dark attack, so in the mirror an active Munkidori (110 HP) eats the
    opposing Grimmsnarl ex's Shadow Bullet doubled for 180*2=360 -- an easy free prize for
    no reason. Retreat cost is only 1 (cheap). So: if the active is Munkidori AND the
    bench has a real attacker from the Marnie line (Impidimp/Morgrem/Grimmsnarl ex,
    646/647/648) above 30% HP to promote instead, retreating is a strict improvement over
    sitting there -- matchup-independent (this is bad everywhere, not just some archetype).
    Falls through to the old flat -100 whenever there's nothing better to promote (bench
    empty, or only fodder/low-HP mons) -- never retreats into a worse spot.

    [майнинг v062] item 4 -- VERIFIED against the expert data, no logic change: playbook
    docs/gold/v062_expert_playbook_lines.md lines 20-24 report median_hp_pct_at_retreat ==
    100.0 in ALL 8 matchup x outcome buckets -- experts never wait for HP pressure to
    retreat/switch, it is a scheduled tempo action, not a "flee at low HP" reaction ("Do not
    code a HP-triggered retreat heuristic"). The 30% check above reads the BENCH TARGET's
    HP (is there something healthy enough worth promoting), never the retreating Munkidori's
    own HP, so this guard was already tempo-scheduled, not HP-triggered on the retreater --
    matches the mined evidence as-is. The playbook's other speculative idea ("protective
    retreat of Grimmsnarl under a big incoming attack") was deliberately NOT implemented,
    per the same evidence."""
    active = active_pokemon(obs)
    if active is not None and active.id == MUNKIDORI:
        for p in my_state(obs).bench:
            if not p or p.id not in MARNIE_LINE:
                continue
            max_hp = getattr(p, "maxHp", p.hp) or p.hp
            if max_hp > 0 and (p.hp / max_hp) > 0.30:
                return 6000, "retreat useless active Munkidori"
    return -100, "avoid retreat"


_MAIN_DISPATCH = {
    OptionType.PLAY: score_play, OptionType.EVOLVE: score_evolve,
    OptionType.ATTACH: score_attach, OptionType.RETREAT: score_retreat,
}


# ── score_target (ATTACH_TO/ATTACH_FROM, TO_FIELD/TO_BENCH, HEAL, DAMAGE,
#                  SWITCH/TO_ACTIVE) ────────────────────────────────────

def score_target(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ctx = obs.select.context

    if ctx in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
        # These two contexts can present either "which of my Pokemon receives the
        # energy" (a Pokemon-in-play object, has .hp) or "which energy card do I
        # attach" (a plain Card, e.g. from Punk Up's multi-energy search). Duck-type
        # on the returned object rather than hardcoding by context name, since which
        # context carries which shape is an engine-flow detail, not something to
        # guess at from the context's label alone.
        if _is_pokemon_card(card):
            if d_energy_count(card) >= 3:
                return -5000, "skip: 3+ Dark energy already"
            area = opt.area if opt.area is not None else opt.inPlayArea
            return attach_target_score(card, area), "attach: target Marnie mon"
        return (5000, "attach: Dark energy") if cid == D_ENERGY else (1000, "attach: card")

    if ctx in (SelectContext.TO_FIELD, SelectContext.TO_BENCH):
        return {
            GRIMMSNARL: (18000, "put Grimmsnarl ex into play"),
            MORGREM: (16000, "put Morgrem into play"),
            IMPIDIMP: (14000, "put Impidimp into play"),
            MUNKIDORI: (13000, "put Munkidori into play"),
            FROSLASS: (8000, "put Froslass into play"),
            SNORUNT: (6000, "put Snorunt into play"),
        }.get(cid, (1000, "put into play"))

    if ctx == SelectContext.HEAL:
        # No heal card in this deck; harmless default (never actually selected).
        return damage_on(card), "heal"

    if ctx == SelectContext.DAMAGE:
        # Shadow Bullet's +30 bench splash target: finish the lowest-HP opp mon.
        hp = getattr(card, "hp", 999) if card else 999
        return 10000 - hp, "damage: lowest HP"

    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        # Own-side SWITCH/TO_ACTIVE never reaches here -- the promotion guard in
        # main.py intercepts those first. Only the opponent-side (Boss gust target)
        # and any stray non-promotion case arrive here.
        yi = obs.current.yourIndex
        pi = getattr(opt, "playerIndex", yi)
        if pi != yi and card:
            pv = prize_value(card)
            te = energy_count(card)
            dmg = boss_attacker_damage(obs)
            killable = dmg > 0 and effective_damage(dmg, card) >= card.hp
            if killable:
                return 20000 + pv * 3000 + te * 100, "Boss: KO"
            # [майнинг v062] item 3: same matchup-specific playbook-target bonus as
            # score_boss's own target estimate (_boss_target_key/_boss_tier_bonus, table
            # BOSS_TARGET_PRIORITY) -- this is the branch that actually picks the gust
            # target, so the bonus has to live here too, not just in score_boss's PLAY-time
            # estimate. Replaces the wave-1 tr_mewtwo/Spidops-only flat bonus. Also folds in
            # the universal "prefer an already-damaged target" bonus (playbook line 68-70).
            matchup = detect_matchup(obs)
            tier_bonus = _boss_tier_bonus(matchup, card.id)
            damage_bonus = min(1000, damage_on(card) * 5)
            if tier_bonus > 0:
                reason = f"Boss: drag playbook target ({matchup}: id={card.id})"
            else:
                reason = "Boss: drag scariest threat"
            return 5000 + tier_bonus + damage_bonus + pv * 1000 + te * 200, reason
        return 1000, "generic promote (guard should win)"

    return 1000, "generic target"


# ── score_to_hand (TO_HAND) ───────────────────────────────────────────────

def score_to_hand(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    effect = getattr(obs.select, "effect", None)
    effect_id = effect.id if effect else None

    if effect_id == POKE_PAD:  # non-Rule-Box Pokemon (648 never offered -- it's ex)
        # [v062 wave3 fix] Munkidori must physically reach the bench by turn 1-2
        # (expert norm; was medianing turn 5 -- out/v062_local_loss_forensic.md Q1).
        # Individual per-card "is it already in play" checks now replace the old
        # flat Morgrem/Munkidori priorities and the mismatched shared "line
        # incomplete" flag on Impidimp (that flag used to fire off Munkidori's
        # absence, not Impidimp's, causing redundant 2nd-copy Impidimp fetches).
        if cid == IMPIDIMP:
            if not attacker_line_started(obs):
                return 20000, "Poke Pad: Impidimp (starting attacker)"
            # [v062 wave3 fix, round 2] excess copies still have real rebuild value
            # (repeated Rare Candy -> Grimmsnarl ex after KOs -- v044/Archaludon
            # forensic Q3 measured n_grimm_evolve=2.27/game) -- keep them mid-tier
            # rather than bottom-of-barrel, just below Munkidori/Morgrem.
            return 8000, "Poke Pad: Impidimp (excess copy, line started)"
        if cid == MUNKIDORI:
            if not has_in_play(obs, MUNKIDORI):
                return 19000, "Poke Pad: Munkidori (not yet in play -- priority)"
            return 16000, "Poke Pad: Munkidori (already in play)"
        if cid == MORGREM:
            return 18000, "Poke Pad: Morgrem"
        if cid in (FROSLASS, SNORUNT):
            return 9000, "Poke Pad: tech line"
        return 1000, "Poke Pad: generic"

    if effect_id == SPIKEMUTH:  # Marnie's Pokemon
        if cid == GRIMMSNARL:
            return 20000, "Spikemuth: Grimmsnarl ex"
        if cid == MORGREM:
            return 17000, "Spikemuth: Morgrem"
        if cid == IMPIDIMP:
            return 15000, "Spikemuth: Impidimp"
        return 1000, "Spikemuth: generic"

    if effect_id == PETREL:  # search a Trainer
        if cid == BOSS and lethal_gust_exists(obs):
            return 20000, "Petrel: Boss (lethal gust)"
        if cid == RARE_CANDY and can_evolve_to_grimmsnarl_now(obs):
            return 18000, "Petrel: Rare Candy (ready evolve)"
        if cid == POFFIN and bench_thin(obs):
            return 15000, "Petrel: Poffin (bench thin)"
        if cid == NIGHT_STRETCHER and key_mon_in_discard(obs):
            return 14000, "Petrel: Night Stretcher (key mon in discard)"
        if cid == LILLIE:
            return 10000, "Petrel: Lillie (refill)"
        return 1000, "Petrel: generic"

    if effect_id == POKEGEAR:  # look top 7, take a Supporter
        if cid == BOSS and lethal_gust_exists(obs):
            return 20000, "Pokegear: Boss (lethal)"
        if cid == LILLIE:
            return 12000, "Pokegear: Lillie"
        if cid == DAWN and getattr(obs.current, "turn", 0) <= 4:
            return 11000, "Pokegear: Dawn"
        return 8000, "Pokegear: supporter"

    if effect_id == NIGHT_STRETCHER:  # Pokemon or Basic Energy from discard
        if cid == GRIMMSNARL:
            return 20000, "Night Stretcher: Grimmsnarl ex"
        if cid == MORGREM:
            return 16000, "Night Stretcher: Morgrem"
        if cid == IMPIDIMP:
            return 12000, "Night Stretcher: Impidimp"
        if cid == D_ENERGY and energy_starved(obs):
            return 10000, "Night Stretcher: Dark energy (starved)"
        return 1000, "Night Stretcher: generic"

    if effect_id == DAWN:  # one Basic + one Stage1 + one Stage2 (3 separate picks)
        # [v062 wave3 fix] same dynamic-priority pattern as POKE_PAD above: Munkidori
        # only needs to beat an *excess* 2nd+ Impidimp, not the starting one.
        if cid == GRIMMSNARL:
            return 20000, "Dawn: Grimmsnarl ex (Stage2)"
        if cid == MORGREM:
            return 17000, "Dawn: Morgrem (Stage1)"
        if cid == IMPIDIMP:
            if not attacker_line_started(obs):
                return 18000, "Dawn: Impidimp (Basic, starting attacker)"
            return 8000, "Dawn: Impidimp (Basic, excess copy)"
        if cid == FROSLASS:
            return 9000, "Dawn: Froslass (Stage1)"
        if cid == MUNKIDORI:
            if not has_in_play(obs, MUNKIDORI):
                return 15000, "Dawn: Munkidori (Basic, not yet in play -- priority)"
            return 12000, "Dawn: Munkidori (Basic)"
        if cid == SNORUNT:
            return 8000, "Dawn: Snorunt (Basic)"
        return 1000, "Dawn: generic"

    # Generic TO_HAND fallback (unlisted / unrecognized effect).
    # [v062 wave3 fix] same dynamic-priority pattern as POKE_PAD/DAWN above.
    if cid == GRIMMSNARL:
        return 18000, "take Grimmsnarl ex"
    if cid == MORGREM:
        return 16000, "take Morgrem"
    if cid == IMPIDIMP:
        if not attacker_line_started(obs):
            return 15000, "take Impidimp (starting attacker)"
        return 8000, "take Impidimp (excess copy)"
    if cid == MUNKIDORI:
        if not has_in_play(obs, MUNKIDORI):
            return 17000, "take Munkidori (not yet in play -- priority)"
        return 12000, "take Munkidori"
    return 1000, "generic take"


# ── score_discard (DISCARD / DISCARD_CARD_OR_ATTACHED_CARD) ──────────────

def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)

    if cid == D_ENERGY:
        if ids.count(D_ENERGY) > 2:
            return 12000, "discard extra Dark energy"
        return -1000, "keep Dark energy"
    if cid in (GRIMMSNARL, MORGREM, IMPIDIMP, MUNKIDORI):
        return -5000, "keep line Pokemon"
    if cid in (BOSS, XEROSIC, PETREL, POKEGEAR):
        return 8500, "discard utility"
    if cid in SUPPORTERS and ids.count(cid) > 1:
        return 8000, "discard duplicate supporter"
    if cid in ITEMS and ids.count(cid) > 1:
        return 7500, "discard duplicate item"
    return 1000, "generic discard"


# ── NUMBER / ACTIVATE ─────────────────────────────────────────────────────

def score_number(obs, opt):
    n = opt.number if opt.number is not None else 0
    effect = getattr(obs.select, "effect", None)
    effect_id = effect.id if effect else None
    if effect_id == GRIMMSNARL:
        # Punk Up's "search up to 5 Basic {D} Energy" count. OPEN ITEM per DESIGN.md
        # Phase 2: this anti-deckout cap is a best-effort placeholder, not verified
        # against a real Punk-Up-with-low-deck trace (none was observed in the local
        # smoke test below). Below deckCount<=8, taper preference for smaller counts
        # rather than blindly maxing out.
        dc = my_state(obs).deckCount
        if dc <= 8:
            safe_max = max(0, dc - 3)
            if n > safe_max:
                return -1000 + n, f"Punk Up: cap count (deckCount={dc})"
    return n, "number"


def score_activate(obs, opt):
    cc = getattr(obs.select, "contextCard", None)
    cc_id = cc.id if cc else None
    is_yes = opt.type == OptionType.YES

    if cc_id == GRIMMSNARL:  # Punk Up
        deck_low = my_state(obs).deckCount <= 8
        if is_yes:
            return (-100000, "Punk Up: skip (anti-deckout)") if deck_low else (100000, "Punk Up: fire")
        return (100000, "Punk Up: decline (anti-deckout)") if deck_low else (-100000, "Punk Up: never decline")

    if cc_id == FROSLASS:  # Freezing Shroud / other passive triggers -- upside but low-stakes
        return (1, "Freezing Shroud: fire") if is_yes else (-1, "Freezing Shroud: decline")

    return (100000, "activate ability") if is_yes else (-100000, "never decline")


# ── score_option dispatch + choose_options ────────────────────────────────

def score_option(obs, opt):
    ctx = obs.select.context
    # [форензик v062] item 2: refresh the sticky matchup label every decision (cheap board
    # scan, called once per option -- see detect_matchup docstring). This is what keeps
    # `_last_matchup` current even on decisions/branches that never read the matchup
    # themselves (e.g. a turn with no Boss's Orders in hand), so it's already fresh the
    # moment a branch that DOES need it (score_boss / score_target item 5) is reached.
    detect_matchup(obs)

    if ctx in {SelectContext.IS_FIRST, SelectContext.MULLIGAN,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON}:
        return score_setup(obs, opt)

    if opt.type in {OptionType.YES, OptionType.NO}:
        if ctx == SelectContext.IS_FIRST:
            return score_setup(obs, opt)
        if ctx == SelectContext.ACTIVATE:
            return score_activate(obs, opt)
        return (1, "yes") if opt.type == OptionType.YES else (0, "no")

    if opt.type == OptionType.NUMBER:
        return score_number(obs, opt)

    if ctx == SelectContext.MAIN:
        fn = _MAIN_DISPATCH.get(opt.type)
        if fn:
            score, reason = fn(obs, opt)
        elif opt.type == OptionType.ABILITY:
            score, reason = 1, "ability"
        elif opt.type == OptionType.ATTACK:
            score, reason = score_attack(obs, opt)
        elif opt.type == OptionType.END:
            score, reason = 0, "end turn"
        else:
            score, reason = 500, "generic MAIN"
    elif ctx == SelectContext.EVOLVE:
        score, reason = score_evolve(obs, opt)
    elif ctx == SelectContext.TO_HAND:
        score, reason = score_to_hand(obs, opt)
    elif ctx in {SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}:
        score, reason = score_discard(obs, opt)
    elif ctx in {SelectContext.ATTACH_TO, SelectContext.TO_FIELD, SelectContext.TO_BENCH,
                 SelectContext.ATTACH_FROM, SelectContext.SWITCH, SelectContext.TO_ACTIVE,
                 SelectContext.HEAL, SelectContext.DAMAGE}:
        score, reason = score_target(obs, opt)
    elif ctx == SelectContext.ATTACK:
        score, reason = score_attack(obs, opt)
    elif ctx == SelectContext.SKILL_ORDER:
        idx = opt.index if opt.index is not None else 0
        score, reason = -idx, "skill order: ascending index"
    elif ctx == SelectContext.DISCARD_ENERGY:
        score, reason = 1000, "discard energy: flat"
    elif opt.type == OptionType.CARD:
        score, reason = score_to_hand(obs, opt)
    elif opt.type == OptionType.ENERGY:
        score, reason = 1000, "energy"
    elif opt.type == OptionType.END:
        score, reason = 0, "end"
    else:
        score, reason = 100, "fallback"

    return apply_overrides(obs, opt, score, reason)


# ── Per-decision trace logging (for post-hoc "why this move" analysis) ────────
# Kaggle captures per-move stdout/stderr in the downloadable agent logs
# (episode-{id}-agent-{n}.json) — verified live for v047. Each scorer decision emits
# ONE compact line to stderr with the winning option's reason+score and the runner-up
# (the decision margin), so a downloaded game can be replayed move-by-move. Never throws;
# toggle DECISION_LOG=False to silence (e.g. if a future run hits the 600s/episode budget).
DECISION_LOG = True


def _dlog(msg):
    if DECISION_LOG:
        try:
            print(f"[dec] {msg}", file=_sys.stderr, flush=True)
        except Exception:
            pass


def _dlog_alt(msg):
    # [форензик v062] item: full-option-dump companion line to [dec], tagged
    # "[alt]" so a downloaded Kaggle log can grep either independently.
    if DECISION_LOG:
        try:
            print(f"[alt] {msg}", file=_sys.stderr, flush=True)
        except Exception:
            pass


def choose_options(obs):
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            score, reason = score_option(obs, opt)
        except Exception as e:
            score, reason = -999999, f"error {type(e).__name__}: {e}"
        scored.append((score, i, reason))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    for score, i, reason in scored:
        if len(selected) >= obs.select.maxCount:
            break
        if score < 0 and len(selected) >= obs.select.minCount:
            continue
        selected.append(i)

    if len(selected) < obs.select.minCount:
        selected = [i for _, i, _ in scored[:obs.select.minCount]]

    # trace: winning pick + runner-up (decision margin), with turn/context/matchup for replay
    if DECISION_LOG:
        try:
            ctx = int(obs.select.context) if obs.select.context is not None else -1
            turn = getattr(obs.current, "turn", "?")
            matchup = detect_matchup(obs)  # [форензик v062] item 2: tag every trace line
            top = scored[0] if scored else (0, -1, "none")
            second = scored[1] if len(scored) > 1 else None
            sec = f" | 2nd #{second[1]} '{second[2]}' s={second[0]}" if second else ""
            _dlog(f"t{turn} m={matchup} ctx{ctx} n{len(scored)} -> #{top[1]} '{top[2]}' s={top[0]}{sec} "
                  f"pick={selected}")
        except Exception:
            pass

    # trace: full dump of every scored option, sorted by score desc, so a
    # downloaded live log can answer "why NOT option X" (not just the runner-up).
    # Skipped when there's nothing to compare against (single-option decisions).
    if DECISION_LOG and len(scored) > 1:
        try:
            ctx = int(obs.select.context) if obs.select.context is not None else -1
            turn = getattr(obs.current, "turn", "?")
            parts = []
            for score, i, reason in scored:  # already sorted score-desc above
                r = reason if len(reason) <= 40 else reason[:40]
                parts.append(f"#{i} s={score} '{r}'")
            _dlog_alt(f"t{turn} ctx{ctx}: " + "; ".join(parts))
        except Exception:
            pass

    return selected
