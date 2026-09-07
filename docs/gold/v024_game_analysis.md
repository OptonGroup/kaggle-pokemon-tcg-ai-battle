# v024 (card-embedding transformer + RL) — live replay analysis

39 games, submission 54631832. Data: `data/episodes/episode-<id>-replay.json`,
parsed by `scripts/analyze_v024.py`, per-game detail in `data/episodes/v024_analysis.json`.

## Score

**W-L-D = 21-18-0, winrate 53.8%** (39/39 games downloaded and parsed; no ERROR/TIMEOUT
statuses seen — no confirmed engine crashes).

## Winrate by opponent archetype

| Archetype | Games | W | L | WR |
|---|---|---|---|---|
| Mega Lucario ex | 8 | 7 | 1 | 88% |
| Dragapult ex | 6 | 6 | 0 | 100% |
| Starmie | 5 | 2 | 3 | 40% |
| Crustle/Dwebble (mirror) | 4 | 3 | 1 | 75% |
| Alakazam | 2 | 0 | 2 | 0% |
| Archaludon | 2 | 0 | 2 | 0% |
| Grimmsnarl ex/Munkidori | 2 | 1 | 1 | 50% |
| Unknown/rogue (9 distinct lineups, incl. Dudunsparce/Dunsparce, Team Rocket's Articuno/Mewtwo ex, Kyogre, Cinderace/Staryu, Budew/Ethan's Cyndaquil, Hop's Phantump line, Lunatone/Makuhita) | 10 | 2 | 8 | 20% |

**Known-meta archetypes combined: 29 games, 19-10, WR 65.5%.**
**Unrecognized/rogue archetypes combined: 10 games, 2-8, WR 20%.**

Archetype identified by union of opponent's active/bench/discard Pokemon seen across
the whole game, matched against known meta signature names; if no match, labeled
"Unknown" with the 3 most notable Pokemon names.

## Top loss causes (18 losses)

1. **Bad matchup vs specific decks** — 10/18 losses: Starmie (3), Alakazam (2), Archaludon (2),
   Grimmsnarl (1), Crustle mirror (1), Mega Lucario (1). These are decks we also beat elsewhere
   in the sample (except Alakazam/Archaludon, which are 0-2 each) — looks like genuine
   matchup/variance issues, not policy bugs.
2. **Losses to unrecognized/rogue decks — 8/18 losses (44% of all losses)**, against opponents
   outside our known top-meta list (Dudunsparce, Team Rocket subset, Kyogre, Cinderace/Staryu,
   Budew/Ethan's Cyndaquil, Hop's Phantump line). WR against this bucket is only 20% vs 65.5%
   against recognized meta decks. This is the single biggest pattern in the loss data: the NN
   (trained in a league of 6 known archetypes) looks noticeably worse against builds it likely
   never saw in training/self-play.
3. **Never developed a second Pokemon (bench stayed empty all game) — 6/18 losses**
   (episodes 85697867's opener... concretely: c88eea62, 1bfcaef2, 02bdb658, 1234594c, 0b5b5706,
   fc55b60c). Same "mono-Crustle" pattern shows up in 4 wins too (vs Dragapult x3, Starmie x1),
   so bare-bench isn't fatal per se — it wins the race against fast aggro (Dragapult 3/3) but
   loses to anything grindier/disruptive (Alakazam, Archaludon, the rogue-deck bucket).
4. **Fast blowouts, turn ≤ 5 — 6/18 losses** (episodes ending turn 2-5: vs Alakazam x2,
   Archaludon, Cinderace/Staryu, Mega Lucario, Starmie). Consistent with Crustle/Dwebble's
   inherent ramp fragility (needs a few turns to evolve + attach enough energy) against decks
   that can pressure or disrupt early.
5. **Hand clog (bench empty + hand ≥8 cards by mid-game) — 18/39 games total, but split
   10 wins / 8 losses** — present in long games generally (both W and L), so on its own this is
   more a symptom of the energy-heavy 60-card list (lots of Basic {G} Energy drawn and not always
   immediately useful) than a clean bug signal. It IS concentrated in the "Unknown/rogue" loss
   bucket though (6 of the 8 rogue losses also show hand clog), reinforcing point 2.

First-player vs second-player split is not a real factor: 11W/19 first (58%) vs 10W/20 second (50%).

## Why we win

- **Attacks whenever available**: across all 39 games there is not one turn where an attack was
  a legal MAIN-menu option and the agent ended the turn without ever opening the attack menu —
  no "declined free damage" cases found.
- **Consistent evolution**: Dwebble→Crustle happened in 37/39 games (all except the two turn-2
  blowout losses where the game ended before evolution was possible).
- **Dominant vs the current top-2 meta decks**: Mega Lucario ex (7-1, 88%) and Dragapult ex
  (6-0, 100%) — 14 of the 21 wins (67%) come from these two matchups alone. This matches the
  Crustle plan's known profile (steady beatdown, Grass-type positioning) against those decks.
- **Mirror match favorable**: Crustle/Dwebble mirror 3-1 (75%), so the transformer at least
  doesn't lose the mirror to the old IL/LightGBM Crustle builds it's replacing.

## Do we actually play Crustle?

**Yes, 100% of the time** — `our_pokemon_names` is a subset of {Dwebble, Crustle} in all 39
games (deck.csv is fixed at battle_start, so the NN cannot pick a different deck; the relevant
question was whether it fails to progress the game plan, and it doesn't structurally — Crustle
gets evolved and put into the active slot in 37/39 games).

## Explicit NN-blender / bug signals

- **The policy never retreats — anywhere, ever.** RETREAT was a legal MAIN-menu option at 94
  separate decision points across 23 of the 39 games (both wins and losses, e.g. long games like
  the 28-turn Crustle mirror loss and 28-turn Crustle mirror win), and it was chosen **0 times**.
  This is a clean, reproducible signature — looks like the RETREAT action head is either
  undertrained or effectively masked/never sampled by the argmax policy. It does not appear to be
  a decisive loss driver by itself (retreat was available in wins just as often as losses,
  e.g. Mega Lucario 4/5 wins had retreat available and unused), but it means the agent can never
  rotate a damaged attacker to a healthy bench Pokemon or dodge a bad matchup by switching — a
  real tactical hole, most costly in the two 28-turn grindy games (Crustle mirror L
  episode 85690473 — 7 missed retreat opportunities across turns 7-25; Grimmsnarl L
  episode 85694312 — 3 missed opportunities).
- **No declined-attack / lethal-skip cases found** (see "Why we win" above) — this specific
  failure mode that was flagged as a risk in the brief does not show up in this sample.
- **No retreat-oscillation loops** detected (the specific "flip-flop between two benched mons"
  pattern was checked for and not found).
- **No ERROR/TIMEOUT statuses** in any of the 39 replays — no evidence of crashes or engine-level
  agent failures.
- **6/39 games are turn-2-to-5 blowout losses** — too fast to see much decision-making at all;
  these read as opening-hand / early-disruption losses rather than mid-game policy failures.

## Conclusions — what to fix to raise live μ

1. **Generalization gap is the single biggest lever.** 65.5% WR vs the known top-meta archetypes
   the model's league/self-play presumably trained against, but only 20% (2-8) vs everything
   else — and "everything else" was a full quarter of the live sample (10/39 games). If the
   live matchmaking pool has a similar ~25% share of non-top-meta/rogue decks, closing even half
   of that gap (20%→50%) would be worth roughly +3pp of overall winrate. Action: widen the RL
   league/self-play opponent pool beyond the current 6 trained archetypes (or add a
   distillation/fine-tune pass on the rogue decks seen here), rather than over-fitting further to
   Lucario/Dragapult where we already dominate.
2. **Teach/patch retreat behavior.** 0-for-94 on a legal action across nearly 60% of games is a
   clear, fixable gap — worth checking whether retreat states are underrepresented in the RL
   training data/reward shaping, or whether the action-space masking silently zeroes out RETREAT
   logits. Even a simple heuristic guard (retreat when active is about to die and a healthier
   attacker is benched) layered on top of the NN could close out some of the longer grindy losses
   (85690473, 85694312).
3. **Look at the 6 turn-≤5 blowout losses concretely** (episodes 85694786, 85692398, 85688091,
   85687130, 85686488, 85683752) — a third of all losses end before turn 5. Worth checking
   whether these are simply bad opening hands (mulligan variance, nothing to fix) or a fixable
   opening-sequencing problem (e.g., not prioritizing the single available Basic/evolution line
   fast enough under pressure).

Sample caveat: 39 games is a small base, especially per-archetype (Alakazam and Archaludon are
each only n=2, both losses — could easily be 0/2 by variance, not a true 0% matchup). The
known-meta-vs-rogue split (29 vs 10 games) is the most statistically meaningful pattern found.
