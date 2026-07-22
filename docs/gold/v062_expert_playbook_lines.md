# v062 expert playbook — numeric lines for coding

Source: `scripts/mine_expert_playbook.py`, mined from `data/episodes/official/{2026-07-14..18,2026-07-20}`
(6 days, 2711 pure jiatu.l/bono games, purity filter >=54/60 deck-overlap). Full aggregates:
`data/il_grimmsnarl_v2/expert_playbook.json`. Per-game raw facts: `data/il_grimmsnarl_v2/playbook_by_day/*.jsonl`.
Archaludon narrative recon: `data/il_grimmsnarl_v2/archaludon_recon/*.md` (6 files, all wins found).

All numbers below are read directly off `expert_playbook.json` unless marked "n мало" (small-sample caveat).
n_games per bucket: crustle 205W/282L, alakazam 595W/505L, mirror 230W/190L, tr_mewtwo 136W/130L,
generic 199W/219L, archaludon 6W/5L (n мало), lucario 6W/3L (n мало).

## Universal rules (hold across every matchup, not matchup-specific)

- **First energy of the game overwhelmingly goes to Munkidori, not Grimmsnarl.** Across every matchup×outcome
  bucket, the ordinal-1 attach target is Munkidori 48-80% of the time (grimmsnarl_line only 20-45%). Median
  turn of Munkidori's first {D} energy is turn 1-2 in every single bucket. This answers directly: attach-routing
  to Munkidori is a **universal opening rule**, not a tank-matchup-specific tactic.
- **Munkidori reliably gets *an* energy at all**: pct_games_munkidori_ever_gets_energy is 90-100% in every
  bucket (lowest: alakazam-lost 90.3%, tr_mewtwo-lost 90.8%).
- **Retreats/switches are essentially never defensive.** median_hp_pct_at_retreat = 100.0 in *every single*
  matchup×outcome bucket (8/8 buckets, both won and lost). The expert never waits for HP pressure to switch —
  switching is a scheduled tempo/rotation action (bring in a fresh attacker, bench Munkidori once its energy
  is placed), not a "flee at low HP" reaction. **Do not code a HP-triggered retreat heuristic** — it would not
  match expert play; code retreat as tempo-scheduled instead.
- **Grimmsnarl ex enters via Rare Candy roughly half the time or more everywhere** (46-77% of the *first* 648
  in a game, matchup-dependent — see below); it is the dominant path to fielding the attacker turn 4-6, not an
  occasional shortcut.
- **Boss's Orders target choice is matchup-specific and clearly non-random** (see per-matchup section) — always
  picking the highest-leverage soft target (a big bench ex, or the opposing Munkidori in mirror), essentially
  never the opponent's own wall/tank if a softer high-value target exists.

## vs Crustle (205W / 282L) — the deck's hardest matchup by win-rate (42% win)

- **Adrena-Brain fires far more in wins**: mean 10.68 activations/game (won) vs 6.86 (lost); pct games with
  >=1 activation 99.0% (won) vs 86.5% (lost). Winning games grind through more Adrena-Brain cycles — Crustle's
  Mysterious Rock Inn blocks ex-attack damage but *not* the Adrena-Brain counter-transfer, so the ability is
  the deck's main answer to the wall. **Code: prioritize Adrena-Brain usage aggressively vs Crustle, every turn
  it's available.**
- **Boss's Orders target priority vs Crustle decks is NOT the wall itself.** Top target by far is card id 756
  = **Mega Kangaskhan ex** (78 pulls in wins, 77 in losses — the single most-pulled target across the whole
  dataset), then 344 = Dwebble (basic pre-evo, 46W/53L), then 345 = Crustle itself (35W/57L, i.e. *less* often
  in wins — sniping Crustle directly is comparatively de-prioritized). **Code: vs Crustle-archetype decks, Boss
  target priority should be Mega Kangaskhan ex > Dwebble > Crustle**, not "always snipe the active wall."
- **Stadium**: played in 93.2% of wins vs only 75.2% of losses (large gap) — median first play turn 3 (won) vs
  4 (lost). Replace-rate (opponent already had a stadium down) 54.9% (won) / 63.7% (lost). **Getting Spikemuth
  down early and reliably correlates strongly with winning this matchup.**
- **Evolve timing**: 648 never appears in 28.4% of *losses* vs only 4.9% of wins — a big signal that failing to
  land Grimmsnarl ex at all is a major loss driver here. Median first-648 turn is 6 in both buckets (so it's not
  about being *late*, it's about *never landing it* — mulligan/hand-clog risk).
- **Munkidori active-duration**: median 4 ticks (won) vs 2 (lost) — Munkidori survives active longer in wins;
  death-rate while active is low either way (2.7% won / 0.8% lost — Crustle rarely punishes Munkidori directly).

## vs Alakazam (595W / 505L, the largest bucket, ~54% win)

- **Adrena-Brain volume is the single starkest differentiator in the whole dataset for this matchup**: mean
  4.21 activations/game (won) vs 2.15 (lost) — almost exactly 2x. pct games with >=1 activation: 94.5% (won)
  vs only 69.1% (lost). **This is the strongest per-matchup correlate of winning found in the mined data.**
- **Adrena-Brain targeting flips zone**: in wins, targets are 65.4% opponent-bench / 15.2% opponent-active; in
  losses, even more bench-skewed (79.5% bench / 9.5% active). So Adrena-Brain vs Alakazam is used almost
  entirely to snipe/finish *bench* targets (see Boss's Orders below — same target priority), not to chip the
  active Alakazam.
- **Boss's Orders target vs Alakazam decks: overwhelmingly card id 140 = Fezandipiti ex** (167 pulls in wins
  out of 238 total = 70% of all Boss uses; only 82/156 = 53% in losses) — targeting rate on Fezandipiti is
  17pp higher in wins. Abra/Kadabra/Alakazam line (741/742/743) is the distant second target (~20%).
  **Code: vs Alakazam-archetype, Boss's Orders priority should be Fezandipiti ex first, Abra/Kadabra/Alakazam
  line second — not the active Alakazam by default.**
- Boss's Orders usage rate itself is also much higher in wins: 38.0% of games use it >=1x (won) vs 26.3% (lost);
  mean 0.40/game (won) vs 0.31 (lost); median target HP% pulled is *lower* in wins (81.0% vs 100% in losses) —
  i.e. winning games pull targets that already have some damage on them (finishing blows), losing games pull
  fresh full-HP targets (less efficient).
- **evolve_grimmsnarl_ex**: 648 never appears in only 1.2% of wins vs 8.1% of losses — again, landing the ex at
  all matters. First-648 via Rare Candy 71.8% (won) vs 65.5% (lost).

## Mirror (Grimmsnarl vs Grimmsnarl, 230W / 190L, ~55% win)

- **Boss's Orders target is almost exclusively the opposing Munkidori**: card id 112 = Munkidori is 63/115
  (55%) of all Boss's Orders uses in wins, 56/67 (84%) in losses — in mirror, pulling the enemy's ability
  engine is by far the dominant use of Boss's Orders (own Grimmsnarl ex 648 is a distant second at 15 uses in
  wins, essentially never targeted in losses).
- **Adrena-Brain volume differentiates wins/losses again**: mean 6.92/game (won) vs 3.55 (lost) — ~2x, same
  pattern as vs Alakazam. Targeting is very bench-heavy in both (74.9% won / 86.9% lost bench-targeted).
- **Stadium play rate is much lower here than any other matchup** (46.1% won / 51.1% lost play a stadium at
  all, vs 90%+ in most other matchups) and replace-rate is near zero (1.9% won / 8.0% lost) — in the mirror,
  both sides run the same Spikemuth Gym, so there's little reason to contest it; the expert mostly just doesn't
  bother playing a second copy over the opponent's.
- **★Munkidori death-while-active rate is the standout signal for this matchup**: 45.2% of Munkidori
  active-spans end in death when the expert *wins*, vs **65.9%** when the expert *loses* — a 21pp gap on a
  reasonably large sample (157 spans won / 167 spans lost). Losing the mirror correlates strongly with your own
  Munkidori getting sniped/KO'd more often while active — i.e. protecting/rotating Munkidori out matters more
  in mirror than anywhere else, since the opponent has the identical Boss's-Orders-into-Shadow-Bullet snipe kit.
- 648 (Grimmsnarl ex) essentially always lands: 0.0% of wins never see it, only 2.6% of losses.

## TR Mewtwo (136W / 130L, ~51% win)

- **Adrena-Brain again ~2x in wins**: mean 3.85/game (won) vs 1.64 (lost); pct games >=1 activation 88.2%
  (won) vs only 59.2% (lost) — the largest won/lost activation-rate gap of any matchup bucket (29pp).
- **Boss's Orders target priority differs sharply between wins and losses.** In wins, 26/41 (63%) of uses
  target 431 = **Team Rocket's Mewtwo ex itself** — makes sense given its Power Saver ability requires 4+ TR
  Pokemon in play before it can attack at all, so sniping it early while it's still "locked" denies it ever
  attacking. In losses, targeting is more spread out toward 414 = TR Articuno (11/43=26%) and 400 = TR
  Tarountula (8/43=19%) — i.e. **losing games waste Boss's Orders on secondary TR support pieces instead of
  the ex itself.** Code: Boss target priority vs TR Mewtwo archetype = the Mewtwo ex (431) first.
- **Stadium replace-rate is high**: 57.3% (won) / 61.6% (lost) of Spikemuth plays are contesting an opponent
  stadium already down — TR decks run their own stadium (Fake ID? — not confirmed by name here) aggressively.
- opp_boss_orders_on_us = 0 in this matchup entirely (both buckets) — TR Mewtwo decks in this sample never used
  Boss's Orders against the expert (or don't run it).

## Archaludon (6W / 5L — **n мало**, do not over-index, but consistent with all other buckets directionally)

- Even at n=6/5, Adrena-Brain fires in 100% of wins vs 80% of losses, mean 4.83 (won) vs 5.6 (lost) — inverted
  vs the pattern elsewhere, but sample too small to trust the direction.
- Boss's Orders target (both W and L, n=2 each) is 169 = Duraludon (the pre-evolution basic), consistent with
  "deny the evolve" logic below.
- Stadium played in 100% of games both sides, turn 1 median — Spikemuth comes down immediately in this
  matchup, replace-rate 71-73% (opponent's own stadium — Full Metal Lab, seen in the recon below — is
  contested hard).
- **See the narrative recon below — this is the strongest finding for this matchup and is not just a stat.**

### Archaludon reconstruction — how jiatu.l/bono actually win the 180-vs-220 race

Read in full: `data/il_grimmsnarl_v2/archaludon_recon/*.md` (6 winning games, full turn-by-turn trace).
Two distinct winning patterns recur across the recon set:

1. **"Never let it evolve."** (clean example: `game_f1f0a758-...md`, a 5-turn blowout.) Duraludon (basic
   pre-evolution, much lower HP than Archaludon ex) sits on the opponent's bench while it powers up with {M}
   energy. Shadow Bullet's built-in 30-damage bench-splash chips it every single attack turn *regardless* of
   who's the active target, and Adrena-Brain's counter-transfer (heal our tank + place counters on their
   weakest revealed target) is aimed at the *same* Duraludon copy rather than the active. Combined chip
   (30 splash + 30 Adrena-Brain, repeated turns 3 and 5, plus a full Shadow Bullet 180 once it's forced active)
   kills Duraludon before it ever evolves into Archaludon ex — the opponent never gets to use Metal Defender at
   all in this game.
2. **"Out-tank the race via redundant bodies + Adrena-recycling."** (clean example:
   `game_4438fef4-...md`, 13 turns.) The expert fields **three separate Grimmsnarl ex bodies** across the game
   (turn 3 via Rare Candy, turn 5 via natural Morgrem evolution, turn 13 via Rare Candy again) — the deck's
   3-copy Grimmsnarl ex + Rare Candy density means a KO'd or heavily-chipped attacker is never a lasting
   setback. Every own-turn, Shadow Bullet fires for 180 into whatever's active (Duraludon or Archaludon ex)
   *regardless* of incoming damage; Metal Defender (220) never one-shots the 320-HP Grimmsnarl ex, and
   Adrena-Brain heals ~30-120 HP back onto the tanking Grimmsnarl every turn *by simultaneously placing that
   damage onto the opponent's backup (not-yet-evolved) Duraludon copies sitting on their bench* — the same
   ability both sustains the tank and preemptively kills the opponent's reinforcements. Boss's Orders (turn 13)
   finishes by pulling the last weak backup Duraludon into active for a final Shadow Bullet. Cumulative Shadow
   Bullet chip (180×3 = 540 across the game, well past Archaludon ex's HP pool) plus denial of backup copies is
   what wins the race — it is not a clean 1-for-1 180-vs-220 trade on a single body.
   
   **Code takeaway**: (a) always fire Shadow Bullet regardless of the incoming Metal Defender threat — the
   bench-splash and cumulative chip matter more than surviving any single exchange; (b) route Adrena-Brain
   damage-placement toward the opponent's *weakest revealed non-active* Pokemon (their backup pre-evolutions),
   not just whatever's active; (c) don't hold back Rare Candy/evolutions to "save resources" — the deck expects
   to burn through 2-3 Grimmsnarl ex bodies per game vs Archaludon.

## Lucario (6W / 3L — **n мало**, directional only)

- Adrena-Brain fires in 100% of games both W/L; mean 5.67 (won) / 6.67 (lost, but median only 2 — one outlier
  game with many activations skews the mean).
- Boss's Orders target both sides is 675 = Lunatone (n=1 won, n=2 lost) — too small to draw a rule from.
- Not enough signal here to code matchup-specific logic; treat as "generic" playbook until more data mined.

## Generic / mixed opponents (199W / 219L)

- Same universal pattern holds: Adrena-Brain mean 4.79 (won) vs 2.54 (lost), activation-rate 93.5% vs 75.8%.
- Boss's Orders targets are diverse (342=Cynthia's Roserade, 90=Thwackey, 381=Cynthia's Garchomp ex, 140=
  Fezandipiti ex, 1071=Meowth ex, 89=Grookey, 1031=Mega Starmie ex...) — no single dominant target, confirming
  this bucket is genuinely a mix of many archetypes rather than one identifiable deck. No matchup-specific rule
  to extract beyond the universal ones above; a generic "target the highest-value/most-damaged revealed
  non-active Pokemon" heuristic is the best fallback.
- Munkidori death-while-active rate: 19.4% (won) vs 22.7% (lost) — same direction as mirror but much smaller
  gap, consistent with it being a real but weaker signal outside true mirror matches.

## Cross-matchup meta-finding: Adrena-Brain volume is the single most consistent win/loss correlate

In every bucket with n>100 games (crustle, alakazam, mirror, tr_mewtwo, generic), mean Adrena-Brain
activations/game is 1.3x-2.6x higher in wins than in losses, and pct-games-with->=1-activation is always
higher in wins (gaps of 12-29 percentage points). This is the most robust, matchup-independent signal in the
whole mined dataset. **If v062 codes only one heuristic change, it should be: maximize Adrena-Brain usage
frequency (never skip an available activation), and route its damage-placement half toward the opponent's
weakest revealed non-active Pokemon by default** (bench-value target, per the Archaludon/Crustle/mirror/
tr_mewtwo target-priority findings above) rather than always hitting the active Pokemon.
