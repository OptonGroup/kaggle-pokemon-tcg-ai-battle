# v061 grimmsnarl_heuristic — design spec (Phase 1+2)

Deterministic matchup-heuristic pilot for the jiatu.l Grimmsnarl/Munkidori deck.
Method = v034/v036 (greedy `score_option → choose_options`), guards = v060 (verbatim + 3 edits),
LightGBM ranker **removed**. Local gate is a regression filter only; live is the arbiter.

## Architecture (two-representation split — CRITICAL)

- **Guards run on the raw `obs_dict`** (integer constants), ported verbatim from v060 `main.py`.
  They run FIRST in `agent()`. Guards cover the OOD/forced contexts the method can't legibly score:
  Adrena-Brain sequence, forced promotion/switch, lethal-KO, energy-attach, Adrena-Brain trigger.
- **Scoring runs on the typed `obs`** = `to_observation_class(obs_dict)` (enums), ported from v036.
  Only reached when NO guard fired. `choose_options(obs)` greedily argmax's `score_option`.
- Both representations come from the same engine dict → option index `i` is identical in both.
  A guard returns indices into `obs_dict["select"]["option"]`; scoring returns indices into
  `obs.select.option`; they map 1:1.

## Files

- `main.py` — path bootstrap + guards (v060, +3 edits) + `to_observation_class` bridge +
  `agent()` never-throw. Imports `scoring_grimmsnarl`. Deletes: `_il_decide`, lightgbm import,
  `features_grimmsnarl_v2` import, pandas/`EN_Card_Data.csv` damage table.
- `scoring_grimmsnarl.py` (NEW) — all v036-style score functions adapted to this deck. Imports
  `cg.api`. Exposes `choose_options(obs)` and `score_option(obs, opt)` (harness imports the latter).
- Delete from bundle: `il_grimmsnarl_v2.txt`, `il_grimmsnarl_v2_JOINT_backup.txt`,
  `features_grimmsnarl_v2.py`, `EN_Card_Data.csv`, `__pycache__/`.

## Deck (jiatu.l, 60 cards) — mechanics

Card taxonomy (cardType: 0=Pokémon,1=Item,3=Supporter,4=Stadium,5=Energy):

Pokémon:
- **648 Grimmsnarl ex** — Stage2 (from Morgrem), HP320, retreat2, weak Grass. Shadow Bullet {D}{D}=180
  (+30 to 1 opp bench, no weakness on bench). Ability **Punk Up**: on evolve-from-hand, search deck
  for up to 5 Basic {D} Energy and attach to your Marnie's Pokémon freely. **WIN-CON.**
- **647 Morgrem** — Stage1 (from Impidimp), HP100, retreat1. Corkscrew Punch {D}{D}=60.
- **646 Impidimp** — Basic, HP70, retreat1. Filch {0}=draw / Corkscrew Punch {D}=10. Line starter.
- **112 Munkidori** — Basic, HP110, retreat1, weak Dark. Mind Bend {P}{C}=60 **UNCASTABLE** (0 {P} in
  deck). Ability **Adrena-Brain**: once/turn, if it has any {D}, move up to 3 counters from 1 of your
  Pokémon to 1 of opponent's. Anti-tank engine. **Needs 1 {D} to switch the ability on.**
- **104 Froslass** — Stage1 (from Snorunt), HP90, retreat1, weak Metal. Frost Smash {W}{C} **UNCASTABLE**
  (0 {W}). Ability **Freezing Shroud**: each Checkup, +1 damage counter on every Pokémon with an Ability
  (both sides) except Froslass → **auto-feeds Adrena-Brain every turn.** Pure passive engine piece.
- **860 Snorunt** — Basic, HP70. Chilly {W} **UNCASTABLE**. Only value = evolve into Froslass.

Only real attackers: **648 (180), 647 (60), 646 (10 or draw)**. 112/104/860 never attack.

`MARNIE_LINE = {646,647,648}` (Punk Up targets; Spikemuth searches these).

Trainers/Stadium/Energy:
- 1086 Buddy-Buddy Poffin (Item) — search 2 Basic ≤70HP to bench. Valid = 646(70),860(70). **NOT** 112(110).
- 1152 Poké Pad (Item) — search a non-Rule-Box Pokémon to hand (648 excluded — it's ex).
- 1227 Lillie's Determination (Supporter) — shuffle hand, draw 6 (8 if exactly 6 prizes). Refill.
- 1219 Team Rocket's Petrel (Supporter) — search a Trainer to hand.
- 1079 Rare Candy (Item) — Basic→Stage2 skip (Impidimp→Grimmsnarl). Not turn1, not just-played basic.
- 1097 Night Stretcher (Item) — a Pokémon or Basic Energy from discard to hand.
- 1080 Unfair Stamp (Item, ACE SPEC) — only legal if you had a KO last opp turn; each shuffles hand,
  you draw 5, opp draws 2. Refill+disruption.
- 1182 Boss's Orders (Supporter) — gust 1 opp benched Pokémon to Active.
- 1231 Dawn (Supporter) — search a Basic + a Stage1 + a Stage2 to hand (assembles the whole line).
- 1197 Xerosic's Machinations (Supporter) — opp discards down to 3 cards.
- 1122 Pokégear 3.0 (Item) — look top 7, take a Supporter.
- 1259 Spikemuth Gym (Stadium) — once/turn each player may search a Marnie's Pokémon to hand.
- 7 Basic {D} Energy ×10.

`ITEMS={1086,1152,1079,1097,1080,1122}`  `SUPPORTERS={1227,1219,1182,1231,1197}`  `D_ENERGY=7`

## Engine facts (verified locally)

- `obs.current` fields: turn, turnActionCount, firstPlayer, yourIndex, result, players, stadium,
  looking, energyAttached, retreated, supporterPlayed, stadiumPlayed. **`obs.current.turn`** = 0,1,2,…
- `obs.select` fields: context, contextCard, deck, effect, maxCount, minCount, option, type,
  remainDamageCounter, remainEnergyCost. Use `obs.select.effect.id` (like v036) to know which
  card's effect is resolving (search/discard contexts).
- option fields: type, area, index, inPlayArea, inPlayIndex, playerIndex, attackId, cardId, number,
  count, energyIndex, serial, toolIndex, specialConditionType.
- SelectContext values that matter: MAIN=0, SETUP_ACTIVE=1, SETUP_BENCH=2, SWITCH=3, TO_ACTIVE=4,
  TO_BENCH=5, TO_FIELD=6, TO_HAND=7, DISCARD=8, DAMAGE=15, REMOVE_DAMAGE_COUNTER=16, HEAL=17,
  ATTACH_FROM=21, ATTACH_TO=22, DISCARD_ENERGY=30, SKILL_ORDER=34, ATTACK=35, EVOLVE=37,
  DRAW_COUNT=38, REMOVE_DAMAGE_COUNTER_COUNT=40, IS_FIRST=41, MULLIGAN=42, ACTIVATE=43.
- OptionType: NUMBER=0, YES=1, NO=2, CARD=3, ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10,
  RETREAT=12, ATTACK=13, END=14, SKILL=15.
- AreaType: HAND=2, DISCARD=3, ACTIVE=4, BENCH=5, STADIUM=7.

## v036 helpers to port (adapt IDs to this deck)

Keep verbatim: `get_card`, `option_card`, `option_target`, `my_state`, `opp_state`,
`active_pokemon`, `opp_active_pokemon`, `opp_bench_pokemon`, `all_my_pokemon`, `hand_ids`,
`discard_ids`, `energy_count`, `retreat_cost`, `damage_on`, `has_tool`, `count_in_play`,
`has_in_play`, `prize_value`, `_first_option_index`, `CARD_DB`, `ALL_ATTACKS`, opp-attack log
tracking (`_update_opp_attack_tracking`), `read_deck_csv`.

Adapt: weakness/effective-damage helpers. Grimmsnarl weakness type differs — use CardData.weakness
generically (`effective_damage(base,target)= base*2 if target weak to Dark(7) else base`, since our
only real damage source is Dark-type Shadow Bullet/Corkscrew).

`d_energy_count(pk)` = count of id-7 energies attached (mirror v060's dark_energy count).

## Matchup detection (port detect_matchup pattern, new tables)

```
CRUSTLE_LINE = {344, 345}          # Rock Inn wall — Shadow Bullet blocked when 345 active
ARCHALUDON_LINE = {169, 190}       # tank beatdown
ALAKAZAM_LINE = {741, 742, 743}    # our best matchup — Powerful Hand OHKO deck
def detect_matchup(obs): crustle / archaludon / alakazam / generic  (by opp active+bench ids)
```
`opp_has_tank(obs)` = opp board shows any of CRUSTLE_LINE∪ARCHALUDON_LINE (mirror v060 `_opp_has_tank`).

## score_option dispatch (ctx-based; port v036 917-965 shape, minus DISCARD_CARD_OR_ATTACHED_CARD)

Return `(score, reason)`; `choose_options` sorts desc, tie-break -index, keeps positives up to
maxCount, always fills to minCount. Attacking/ability ends turn → those score BELOW development.
End every MAIN path through `apply_overrides(obs,opt,score,reason)`.

### score_setup (ctx SETUP_ACTIVE=1, SETUP_BENCH=2, MULLIGAN=42, IS_FIRST=41) — v036 verbatim shape
- MULLIGAN: NO→10000 else 0.  IS_FIRST: NO→10000 (choose second, default; TUNABLE) else 0.
- SETUP_ACTIVE: `{646:(100000,"active Impidimp line"), 860:(3000,"active Snorunt"),
  112:(1000,"active Munkidori fallback")}.get(cid,(0,"unknown active"))`. **Never Munkidori/Snorunt
  preferred over Impidimp; never a Stage1.**
- SETUP_BENCH: −10000 "never bench during setup" (v036 verbatim — TUNABLE, flag in legibility).

### score_play (MAIN, PLAY=7)
- Basic Pokémon to bench: 646→18000, 112→17500 (engine: Adrena-Brain+Freezing-Shroud synergy),
  860→14000 (enables Froslass).
- Stadium 1259 Spikemuth: 20000 (always-useful toolbox search; skip only if our own already in play →
  −200).
- Items default 20000 with point vetoes:
  - 1097 Night Stretcher: −500 unless discard has a Pokémon OR a Basic {D} energy to retrieve
    (retrieve KO'd 648/647 or energy when energy-starved).
  - 1086 Poffin: −500 if bench is full (≥5) OR no valid ≤70 basic would help (already have 646+860
    developed and bench wide); else 20000. (Munkidori 110 is never a Poffin target.)
  - 1079 Rare Candy: gate on `can_evolve_to_grimmsnarl_now(obs)` (an Impidimp 646 in play that was NOT
    put down this turn, AND a Grimmsnarl ex 648 in hand, AND turn≥1). True→21000; else −500 save.
  - 1152 Poké Pad: 20000 if the line is incomplete (need Impidimp/Morgrem/Munkidori and none in hand);
    else 6000.
  - 1122 Pokégear: 16000 if no Supporter in hand and supporter not yet played; else 2000.
  - 1080 Unfair Stamp: 19000 whenever offered (engine only offers it when legal = we took a KO).
- Supporters (all: −1000 if `supporterPlayed`):
  - 1182 Boss: see Boss logic below.
  - 1197 Xerosic: 6000 if `opp.handCount≥4` else −500 save.
  - 1219 Petrel: 6000 if we have a concrete Trainer need (need Boss for a lethal gust & Boss not in
    hand; OR need Rare Candy for a ready Grimmsnarl evolve & not in hand; OR bench thin & no Poffin in
    hand); else 2500.
  - 1227 Lillie: −500 save if Boss in hand AND an attacker is ready (mirror v036 734-739); else 5000
    refill. Low-hand bonus: if handCount≤2 → 9000.
  - 1231 Dawn: 8000 if `obs.current.turn≤4`; else −500 (dead tutor late).
- default → 1000 "generic play".

**Boss (1182) logic** (adapt v036; our attacker = active Grimmsnarl w/ ≥2 {D} → Shadow Bullet 180,
weakness-adj; 0 if opp active is Rock-Inn Crustle 345):
```
if supporterPlayed: return -1000
attacker_dmg = 180 (weakness-adj) if active is 648 with ≥2 {D} else 0   # + consider Corkscrew 60 if 647
if attacker_dmg<=0: return -500, "save Boss: no attacker"
remaining = len(my prize)
for t in opp_bench:
    if effective_damage(attacker_dmg,t) >= t.hp:            # a bench mon we could KO if gusted
        if prize_value(t) >= remaining: return 20000, "LETHAL Boss"
# tempo-drag the scariest fresh threat (83% of real usage — plan fact 4):
best = max over opp_bench of (2000 + prize_value(t)*1000 + energy_count(t)*100)   # -500 if bench empty
return best
```

### score_evolve (MAIN EVOLVE=9  AND  ctx EVOLVE=37) — same function, v036 822-841 pattern
Target the Active slot first; escalate on energy. `option_target` = the basic being evolved.
- Evolving into **648 Grimmsnarl ex** (from Morgrem or via Rare Candy from Impidimp): the payoff play.
  - target is Active: 28000 (+ 2000 if it already has ≥1 {D}). This triggers Punk Up → whole engine.
  - target on Bench: 20000.
- Evolving into **647 Morgrem**: Active 16000 / Bench 12000.
- Evolving into **104 Froslass** (from Snorunt): 14000 (Freezing-Shroud engine; bench is fine).
- default → 10000 "generic evolution".

### score_attach (MAIN ATTACH=8) + attach_target_score (v036 844-877 pattern)
Only {D} energy (id 7) is attachable. `if obs.current.energyAttached: return -1000 "already attached"`.
`attach_target_score(target, area)` with cap+escalation:
- 648 Grimmsnarl: cap 2. e=0→4000, e=1→7000; if already ≥2 → −5000 (don't over-attach). +1500 if Active.
- 112 Munkidori: cap 1. e=0→6000 (switch Adrena-Brain ON — high value engine enabler); if ≥1 → −3000.
- 647 Morgrem: cap 2. e=0→3000, e=1→5000; ≥2 → −4000.
- 646 Impidimp: cap 1. e=0→2000; ≥1 → −3000.
- 104/860/anything else → −2000 (wasting Dark energy on non-attackers/non-engine).
- HP-ratio tweak (v036): ratio≤0.25 −1500, ≤0.5 −500, else +min(1000, hp//40*100).
Note: the energy_guard already force-attaches {D} onto an active Grimmsnarl with <2 — so scoring's
attach mostly resolves Munkidori-enable / benched-line / 2nd-copy cases.

### ATTACK (MAIN ATTACK=13 and ctx ATTACK=35) — weakness-adjusted damage, below development
- `attack_choice_score(obs, attackId)` = effective_damage(base, opp_active). Base by attackId:
  937 Shadow Bullet=180, 936 Corkscrew(Morgrem)=60, 935 Corkscrew(Impidimp)=10, 934 Filch=0.
- **Rock-Inn override** (in apply_overrides): active 648 attacking active 345 Crustle → Shadow Bullet
  score −5000 (does 0 real damage; don't waste the turn — let Adrena-Brain grind / develop / END).
- **Impidimp special**: prefer Filch (934, draw) over Corkscrew (935, 10) unless Corkscrew is lethal
  (10≥opp_hp) or handCount≥7 (dig). Score Filch=100, Corkscrew=−100 normally, high if lethal.

### RETREAT (MAIN RETREAT=12) → −100 "avoid retreat" (v060 conscious choice; keep flat).

### score_target (ctx ATTACH_TO=22, ATTACH_FROM=21, TO_FIELD=6, TO_BENCH=5, HEAL=17, DAMAGE=15,
###                SWITCH=3, TO_ACTIVE=4) — port v036 1072-1132, adapt
NOTE own-side SWITCH/TO_ACTIVE never reaches here (the promotion guard intercepts). Only the
opponent-side (Boss gust target, playerIndex≠me) and the non-promotion contexts arrive.
- ATTACH_TO (Punk Up energy placement + normal attach targeting): `attach_target_score(card, area)` —
  same deficit logic. Positive while a Marnie mon still wants {D}, negative once capped → the
  choose_options top-K loop naturally stops attaching (implements "expert took 2 of 4", plan fact 3).
- ATTACH_FROM: skip 3+ energy target (−5000); else `attach_target_score(card, area)`.
- TO_FIELD/TO_BENCH (put searched Pokémon into play): 648→18000, 647→16000, 646→14000, 112→13000,
  104→8000, 860→6000, else 1000.
- HEAL: `damage_on(card)` (no heal card in deck; harmless default).
- DAMAGE (Shadow Bullet's +30 bench splash target): `10000 - hp` → finish the lowest-HP opp mon
  (v036 verbatim).
- SWITCH/TO_ACTIVE with playerIndex≠me (Boss gust target): killable (Shadow Bullet 180 weakness-adj ≥
  hp)? → 20000 + prize_value*3000 + energy*100; else drag → 5000 + prize_value*1000 + energy*200
  (drag the scariest fresh threat). playerIndex==me fallthrough → generic promote (safety; guard wins).

### score_to_hand (ctx TO_HAND=7) — pick the most-needed searched card, by `obs.select.effect.id`
- effect 1152 Poké Pad (non-RuleBox Pokémon): 646 if line incomplete→20000, 647→18000, 112→16000,
  104/860→9000, else 1000. (648 not offered — Rule Box.)
- effect 1259 Spikemuth (Marnie's Pokémon): 648→20000 (only way besides Dawn/Night-Stretcher to grab
  the ex to hand), 647→17000, 646→15000.
- effect 1219 Petrel (Trainer): 1182 Boss if a lethal gust exists→20000; 1079 Rare Candy if a ready
  Grimmsnarl evolve→18000; 1086 Poffin if bench thin→15000; 1097 Night Stretcher if key mon in
  discard→14000; else 1227 Lillie→10000; else first→1000.
- effect 1122 Pokégear (Supporter): 1182 Boss if lethal→20000; else 1227 Lillie→12000; 1231 Dawn if
  turn≤4→11000; else first supporter→8000.
- effect 1097 Night Stretcher (Pokémon/Energy from discard): 648→20000, 647→16000, 646→12000,
  7 (energy) if energy-starved (no {D} on board, none in hand)→10000, else 1000.
- effect 1231 Dawn (one Basic + one Stage1 + one Stage2, i.e. 3 separate picks): among Basics
  646→high; among Stage1 647→high (104 lower); Stage2 648→high. Score by that line preference.
- generic TO_HAND fallback: 646/647/648 line high (18000/16000/15000), 112→12000, else 1000.

### score_discard (ctx DISCARD=8, TO_DECK, etc.) — rarely fires for us; safe default
Keep line Pokémon (648/647/646/112 → negative to keep), discard duplicate items/surplus energy →
positive. Low-stakes; port v036 1027-1069 shape trimmed to our cards.

### NUMBER (OptionType.NUMBER=0) / DRAW_COUNT (ctx 38) / DAMAGE_COUNTER_COUNT
- Default: `opt.number` (pick the max) — matches v036. Adrena-Brain count (ctx 40) is guard-handled.
- **OPEN ITEM (verify before ship, plan §Phase2)**: trace one real DRAW_COUNT (ctx 38) decision to
  confirm max-draw is always correct (no anti-deckout non-max case). If Punk Up presents a count-of-
  energy NUMBER while deckCount≤8, cap it below deckout risk.

### ACTIVATE (ctx 43) — YES/NO to fire an ability (Punk Up, and others)
- Punk Up (contextCard/effect = 648 evolve): YES→100000 UNLESS `my deckCount≤8` (anti-deckout: Punk Up
  removes up to 5 {D} from deck) → then NO. (Plan §Phase2.)
- Freezing Shroud / other passive triggers: YES→1 (fire, harmless) unless a specific reason not to.
- default ACTIVATE: YES→100000, NO→−100000 (mirror v036 Explosiveness pattern — abilities are upside).

### SKILL_ORDER (ctx 34) → lowest option index (ascending, plan).
### DISCARD_ENERGY (ctx 30) → flat 1000 (all {D} identical, plan).
### YES/NO (other contexts) → YES 1 / NO 0 (v036 default).
### Any unhandled ctx/type → 100 "fallback" (never throws; choose_options catches per-option).

## apply_overrides(obs, opt, score, reason)
- Hard: `deckCount≤8` and PLAY of a draw supporter that would over-draw into deckout → deprioritize
  (mirror v036 Explorer guard; here Lillie with tiny deck: keep v036's `dc≤3 & handCount≥dc+6 → refill`
  exception).
- Rock-Inn: active 648 ATTACK vs active 345 Crustle → −5000 (Shadow Bullet does 0). Only override
  gated on matchup; every other matchup untouched (Alakazam/generic win rates structurally preserved).

## Phase-1 guard edits (in main.py, on the v060 guards)
1. `_promo_priority`: Froslass 104 → its own tier between tier1 and tier0 ("tanky utility": survives,
   feeds Freezing Shroud, but can't attack). Snorunt 860 → tier0-ish (fragile). Remove Morpeko 649
   (not in deck — dead code). Net order: 648 > {647 real attacker} > 104 (tanky utility) > unlisted >
   {646,112,860,305,66 fragile/support}.
2. `_adrena_trigger_guard` branch (c): **remove the `_opp_has_tank` gate** — fire the opportunistic
   Adrena-Brain whenever there's damage to siphon (available_snipe>0), any matchup. Rationale:
   Freezing Shroud generates counters every turn regardless of opponent; strict-upside heal-them-
   damage-you transfer never carries over unused. Cheapest, best-justified edit.
3. `_ko_guard`: replace pandas/`EN_Card_Data.csv` `_DAMAGE_TABLE` with a table built from
   `CARD_DB`/`ALL_ATTACKS` (`all_card_data()`/`all_attack()`): `{cardId: [attack.damage for attackId in
   CardData.attacks]}` — CardData.attacks already excludes abilities (skills are separate), so no
   ability-row filtering needed. Keep the Rock-Inn zeroing (648 vs active 345) verbatim.

## v062 changes

Two edit waves on top of v061. Wave 1 = **[форензик]** (diagnosis from v061's own live losses,
2026-07-21). Wave 2 = **[майнинг]** (mined from 2711 pure jiatu.l/bono expert games,
`docs/gold/v062_expert_playbook_lines.md`, full aggregates in
`data/il_grimmsnarl_v2/expert_playbook.json`) — some wave-1 decisions were revised once real
expert-play numbers came in. Both files tag every edit `[форензик v062]` / `[майнинг v062]` at
its own definition.

### Wave 1 — [форензик] (live-loss diagnosis)
- `reset_game_state()` bug fix (`scoring_grimmsnarl.py`): was called by `main.py` on every
  new-game boundary but never *defined* — silent no-op, scorer state leaked across games in one
  Kaggle process.
- Sticky board-scan matchup detection: `_detect_matchup_raw`+`_MATCHUP_PRIORITY_RAW` (`main.py`)
  and `detect_matchup`+`_detect_matchup_board_only` (`scoring_grimmsnarl.py`) — priority-ordered
  crustle>archaludon>lucario>tr_mewtwo>abomasnow>mirror>alakazam>generic, ported from
  v049_bench_sticky_guard's board+sticky layer (no torch/belief-net dependency).
- `_munkidori_energy_guard` (NEW, `main.py`): force the turn's one manual {D} energy attach onto
  a 0-energy Munkidori instead of letting `_energy_guard` always claim it for Grimmsnarl ex —
  diagnosed from 3/4 live Crustle losses where Adrena-Brain fired 0 times all game. Originally
  gated to grind/tank matchups (`_MUNKIDORI_ENERGY_GATE_MATCHUPS`) — **gate removed in wave 2**.
- Active Munkidori retreat (`score_retreat`, `scoring_grimmsnarl.py`): retreating a useless,
  Dark-weak active Munkidori for a real bench attacker scored a flat -100 in v060/v061 (never
  retreat); now a strict-improvement +6000 whenever a Marnie-line mon above 30% HP is benched.
- Boss's Orders vs tr_mewtwo (`score_boss`/`score_target`, `scoring_grimmsnarl.py`, wave-1
  version): prioritized Team Rocket's Spidops (401, the deck's real attacker) over Mewtwo ex
  (mostly inert behind Power Saver) as the gust target — **superseded by wave 2's table** below
  (mined data showed the ex itself is the better target, not Spidops).
- Spikemuth Gym replace-bonus (`score_play`, `scoring_grimmsnarl.py`): playing over a live
  opponent stadium scores above a routine first play (Lucario/Gravity Mountain prep).

### Wave 2 — [майнинг] (2711-game expert mining, revises/extends wave 1)
1. **Munkidori-first energy — matchup gate REMOVED** (`_munkidori_energy_guard`, `main.py`).
   Playbook lines 14-19: the expert sends the game's FIRST energy to Munkidori in EVERY matchup
   bucket (48-80% of ordinal-1 attaches, alakazam/mirror/generic included), and the starkest
   won/lost Adrena-Brain-volume gap in the whole dataset (lines 55-57) is actually vs Alakazam
   (4.21W vs 2.15L activations/game) — the matchup the wave-1 gate excluded. The guard now fires
   in every matchup; its 0-energy self-limit is the only remaining restriction.
2. **Adrena-Brain target routing — bench-by-default for the opportunistic branch**
   (`_adrena_sequence_guard`'s "await_target" step, `main.py`). New `_adrena_intent` global
   (`"kill"|"combo"|"value"`, set by `_adrena_trigger_guard`) lets the guaranteed-KO and
   KO-combo branches keep targeting the active (unchanged), while the opportunistic branch now
   routes to the opponent's bench: prefer a bench mon the transfer KOs outright (lowest remaining
   HP among those), else the single weakest bench mon, else fall back to active. Playbook lines
   58-61 (vs Alakazam: 65.4%/79.5% of Adrena-Brain targets are opponent-bench in won/lost games)
   and lines 124-149 (Archaludon reconstruction: damage routed onto not-yet-evolved backup
   Duraludon copies on the bench — "kill the reinforcement before it evolves").
3. **Boss's Orders — matchup-specific target-priority tables** (`BOSS_TARGET_PRIORITY`,
   `_boss_tier_bonus`, `scoring_grimmsnarl.py`), replacing wave 1's tr_mewtwo/Spidops-only flat
   bonus: crustle → Mega Kangaskhan ex(756) > Dwebble(344) > Crustle(345) itself (playbook lines
   39-43 — the wall is *de*-prioritized in wins); alakazam → Fezandipiti ex(140) > Abra/Kadabra/
   Alakazam line (lines 62-66); tr_mewtwo → TR Mewtwo ex(431) itself FIRST, Spidops(401) second
   (lines 97-102 — flips the wave-1 order: sniping the still-locked-behind-Power-Saver ex denies
   it ever attacking); archaludon → Duraludon(169) (line 112). mirror deliberately has no table
   entry — Munkidori(112) is already covered by the killable branch, and the data shows
   over-fixating on it correlates with *losses* (84% of Boss uses target it in losses vs 55% in
   wins), so no extra bonus was added on top. Plus a small universal bonus for already-damaged
   targets (line 68-70: median HP% of the pulled target is 81% in wins vs 100% in losses).
4. **Retreat — verified, no change.** Playbook lines 20-24: median HP% at retreat is 100.0 in
   all 8 matchup×outcome buckets — the expert never waits for HP pressure to retreat, it's a
   scheduled tempo action. Wave 1's Munkidori retreat already reads the *bench target's* HP, not
   the retreating Munkidori's own HP, so it was already tempo-scheduled, not HP-triggered — no
   change needed. The playbook's speculative "protective retreat of Grimmsnarl under a big
   incoming attack" idea was NOT implemented, per this same evidence against HP-triggered
   retreats.
