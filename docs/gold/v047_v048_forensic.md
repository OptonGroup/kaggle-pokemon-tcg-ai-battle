# v044_resub / v046_judgefix full-corpus forensic (129 live games) → v048 candidates

All 129 live tracked-2 episodes (v044_resub N=68 WR66.2%, v046_judgefix N=61 WR50.8%) analyzed
game-by-game — not a sample, and explicitly not losses-only (see "Methodology" for why). Pipeline:

- `scripts/forensic_v044_v046_full.py` → `data/episodes/v044_v046_forensic.csv` (129 rows),
  `v044_v046_forensic_detail.json` (45 flagged episodes, per-turn timeline),
  `v044_v046_other_deck_clusters.json` (deck signature clusters for the unclassified bucket).
- `scripts/analyze_v044_v046_forensic.py` → `data/episodes/v044_v046_forensic_summary.md` (full
  win-vs-loss cross-tabs, referenced below).

Both reuse `src/replays/parse.py::true_decks_from_data` for ground-truth opponent decks instead of
re-deriving replay parsing.

## Methodology: why every table is split win vs loss

david valor's forum postmortem (topic 713608, 25 votes) reports the "forensic loss-analysis trap":
reading 40 losses found 3 plausible fixes that, on a paired A/B at N≥400, gave a **−7.6pp regression**
— reading losses alone can't distinguish "this pattern causes losses" from "this pattern is just what
losing games look like, and also appears in wins we don't happen to be reading." Every signal below is
therefore reported as a distribution split by **result**, not as a loss narrative; a signal that looks
the same in wins and losses is discarded as noise, not reported as a finding. Any code candidate that
comes out of this is a **hypothesis for a gate (N≥400)**, not a decision.

## Archetype coverage fix (prerequisite for everything else)

The previous ad-hoc matchup breakdown (session-memory `project_v044_v046_live_matchup_breakdown_20260719`)
classified opponents by 8 line-signatures copied from `submissions/v046_judgefix/main.py`'s
`detect_matchup()`, leaving **~40% of games in an undifferentiated "generic/other" bucket** — too
coarse to draw conclusions from. Clustering the 60-card ground-truth decks
(`v044_v046_other_deck_clusters.json`) by "cards appearing ≥2 times" surfaced two large unclassified
clusters:

| Cluster | Signature | Games |
|---|---|---|
| **archaludon_mirror** | Duraludon(169)/Archaludon ex(190)/Cinderace(666) | **19** |
| **dragapult** | Dreepy(119)/Drakloak(120)/Dragapult ex(121) | **14** |

Adding these two shrank "other" from 48/129 (37%) to 15/129 (12%). Neither branch exists in the live
`detect_matchup()` — mechanically identical to the belief-net bug already found in
`project_deckbelief_three_bugs_found.md` ("no prototype for the opponent's mirror of our own
archetype"), except here it's the simple board-signature classifier, not the MLP. **~26% of the live
field (33/129 games) currently gets zero matchup-specific logic** in the shipped agent.

## Finding 1 — v046_judgefix's regression is broader than the prior 2-archetype read

With the corpus properly bucketed (`v044_v046_forensic_summary.md`, "WR by submission × archetype"):

| Archetype | v044 WR (N) | v046 WR (N) |
|---|---|---|
| crustle | 14.3% (7) | 55.6% (9) |
| lucario | 80.0% (5) | 40.0% (10) |
| **archaludon_mirror** | **66.7% (12)** | **42.9% (7)** |
| **dragapult** | **100% (8)** | **50% (6)** |
| alakazam | 55.6% (18) | 53.8% (13) |

The prior read (session-memory, N=56/63, "other" still lumped) concluded the v046 regression was
confined to "generic/other + lucario" while Crustle improved. With mirror and dragapult broken out,
**the same regression pattern appears in two more archetypes that were previously invisible inside
"other."** This changes the interpretation: v046_judgefix's net-negative live result isn't a narrow
side-effect concentrated in 1-2 matchups, it looks like a broad-based decline touching most non-Crustle
opponents — consistent with a **deck-level** cause (affects resource availability regardless of
matchup) rather than a **matchup-conditional code branch** (which would only touch specific archetypes).

## Finding 2 — bench thinness is a real win/loss differentiator, and worse in v046

`bench_thin_frac` (fraction of our own MAIN decisions with bench≤1) is **higher in losses than wins for
both agents** — not noise:

| | v044 win | v044 loss | v046 win | v046 loss |
|---|---|---|---|---|
| mean | 0.384 | 0.476 | 0.444 | 0.590 |
| `frac==1.0` (never bench>1 all game) | 6.7% | 21.7% | 12.9% | 33.3% |

Two independent facts here: (a) within each agent, thin bench correlates with losing (3.2x for v044,
2.6x for v046 on the extreme `==1.0` case) — a genuine W-vs-L signal, survives the anti-forensic-loss-trap
check; (b) **v046 runs thinner benches than v044 in both wins and losses** (12.9% vs 6.7% win, 33.3% vs
21.7% loss) — v046's deck/logic changes measurably thinned the bench across the board, not just in the
matchups it was supposed to fix.

Manually reading two `bench_thin_frac==1.0` v046 losses (episode 86933558, episode 86836918) shows the
mechanism is not uniform:
- **Episode 86836918** (archaludon_mirror, v046 loss): my_prize stayed at 6 the *entire game* (0 KOs
  scored by us) while opp_prize dropped 6→1 by turn 12 — we were simply outpaced in the mirror, never
  landed a knockout, and bench collapse (1→0) tracked that losing tempo rather than causing it. Thin
  bench here is a **symptom** of losing the race, not an independent root cause.
- **Episode 86933558** (lucario, v046 loss): a genuinely anomalous short game — 69 total steps, turn
  stuck at 4 for the last ~40 steps, our own decisions marked `INACTIVE` while the opponent's Mega
  Lucario ability resolution cycles through repeated near-identical prompts, ending abruptly with both
  players mid-way through an unresolved `TO_HAND` select. Bench never built past 0 because the game
  never reached a normal bench-building decision point. This is very likely an **engine/replay-export
  edge case**, not a scoring-logic bug — flagged for a narrow follow-up, not a v048 fix target.

**Conclusion:** bench thinness is a validated correlate of losing and is worse in v046 than v044
(supporting the deck-level-cause hypothesis in Finding 1), but the mechanism is heterogeneous enough
that a blind "always keep bench≥2" heuristic is not obviously safe — any fix needs the same N≥400 gate
discipline, not a direct port.

## Finding 3 — prize-trade deficit by turn 8-12 predicts loss (generalizes prior 3-game read)

`prize_lead_final` sign (my knockouts − opponent knockouts, correct direction confirmed against
`src/cg/api.py:476`) at the final observed frame:

| | v044 win (N=45) | v044 loss (N=23) | v046 win (N=31) | v046 loss (N=30) |
|---|---|---|---|---|
| ahead | 32 (71%) | 4 (17%) | 16 (52%) | 5 (17%) |
| behind | 2 (4%) | 17 (74%) | 4 (13%) | 20 (67%) |

This is the same pattern found manually in 3 v046_final losses
(`project_v046_final_two_losses_bench_wipe.md` — "we proigryvali by razmenam ves' mittel'shpil'"), now
confirmed structurally across the full 53-loss corpus rather than 3 hand-picked games: **being behind on
the knockout trade by turn 8-12 is strongly associated with losing, for both agents.** Not a new lever by
itself (it's a description of losing, not an actionable code change), but it validates that the earlier
manual reads generalized correctly and weren't cherry-picked.

## Finding 4 — Boss's Orders play rate collapses specifically in v046 losses

| | v044 win | v044 loss | v046 win | v046 loss |
|---|---|---|---|---|
| Boss's Orders played by us | 33.3% | 39.1% | 38.7% | **16.7%** |

v046_judgefix plays Boss's Orders in only 16.7% of its losses — roughly half its own win rate (38.7%)
and well below v044's rate in either bucket (33-39%). Consistent with Finding 1/2: in the games v046
loses, it appears to be more resource-starved (fewer supporters played generally), which is what a
`Full Metal Lab` deck-slot cut (traded for a 2nd Judge copy in v046's `deck.csv`) would predict — v046's
`main.py` code changes (`score_discard()` Judge protection, Archaludon ex per-copy cap) don't touch
supporter *access*, only what to discard, so this points at the deck edit rather than the code edit.

## Finding 5 — result_reason breakdown, and a genuine data-export limitation found along the way

`LogType.RESULT` (the API's own machine-readable end-of-game reason field, `src/cg/api.py:321`,
`reason` codes documented at line 320) **never appears in any of the 129 downloaded replays** — verified
by scanning all frames of a 30-file sample, zero hits. Kaggle-environments truncates the observation
stream immediately after the decisive action is submitted, one step before the engine would emit that
log — confirmed concretely on episode 86944821, where the final recorded frame shows the winner still
holding 1 unclaimed prize card with a pending KO-attack log already present, i.e. the true prize count
(0, a sweep win) is never captured in any snapshot. `result_reason` in the CSV is therefore a structural
approximation (winner's own prize-remaining ≤1 → `prize_race`; loser's active+bench both empty →
`no_more_pokemon`; loser's deck empty → `deck_out`; else `unclear`), not a ground-truth field — 61/129
games (47%) land in `unclear`. Within that bucket, v046 has a markedly lower WR (37.5%) than v044
(51.7%) — consistent with everything above, but not independently diagnostic given the label's honesty
limits.

## Finding 6 — matchup-detection lag does NOT differentiate wins from losses

`detect_lag_turn` (first turn our board-only archetype guess matches ground truth) and
`detect_match_frac` are statistically indistinguishable between wins and losses for both agents (median
detect lag = turn 2 in all four W/L × agent buckets; match_frac 0.81-0.88 in all four). **This is the
exact mechanism v047_deckbelief's persistent belief-net was built to fix** (detect_matchup() "forgetting"
an archetype once the identifying Pokémon is discarded) — this dataset gives it no visible win-rate
signal either way. Doesn't contradict shipping v047 (0 regressions on its own gate, and Finding 0 above
shows ~26% of games get zero matchup logic today for an unrelated reason — missing signatures, not
detection lag) — just means it shouldn't be over-weighted as *the* fix for v046→v048's win-rate gap.

## v048 candidates, ranked

**Rejected/closed, not re-litigated here (already settled by prior work, this forensic doesn't change
them):** K4 hand-disruption for our own deck (official-dataset N=2416: Xerosic/Hand Trimmer correlate
with −16.1/−10.4pp), K5 PIMC-supporter-choice (N=400: negative on all 3 matchups), reactive-retreat
(closed 4×), Hero's Cape deck cut (−8.9…−10.5pp vs Crustle, proven). Finding 4's "opponent's Xerosic
against us correlates with our losing" (v044: 34.8% loss vs 15.6% win) is the mirror image of the K4
rejection — it argues for *protecting* our own resources (Judge-protection direction, already in v046),
not adding more disruption of our own.

1. **(highest confidence) Base v048 = v047_deckbelief + port ONLY the `score_discard()` diff from
   v046_judgefix (Judge `-4000 keep` protection + Archaludon ex per-copy cap), WITHOUT the
   `deck.csv` Full Metal Lab → 2nd Judge slot cut.** Findings 1, 2, and 4 independently point at the
   deck-level edit (not the code edit) as the source of v046's broad-based regression, while the code
   edit specifically fixed Crustle (14.3%→55.6-62.5%, reproduced twice now). If the hypothesis holds,
   this should recover the Crustle fix without the archaludon_mirror/dragapult/lucario/generic
   regression. **Requires the already-planned 4-way ablation gate** (deck-only / logic-only / combo /
   baseline, N≥400/cell, new `scripts/v048_judgefix_port_gate.py`) — this forensic raises confidence,
   it does not replace the gate.
2. **Add `archaludon_mirror` ({169,190,666}) and `dragapult` ({119,120,121}) branches to
   `detect_matchup()`.** Currently 33/129 (26%) of the live field falls through to `generic` for a
   structural coverage gap, not a hard matchup — cheap, mechanical, same class of fix as the existing
   `abomasnow`/`hop` branches already in `submissions/v046_judgefix/main.py`. Needs its own gate but is
   low-risk and clearly justified by corpus size (bigger than 6 of the 8 existing branches).
3. **(exploratory, not a direct heuristic port) Bench-thinness mitigation.** Correlates with losing 2.6-
   3.2x at the `bench_thin_frac==1.0` extreme, worse in v046 than v044 — but Finding 2's manual reads
   show the mechanism is heterogeneous (tempo-loss symptom in one case, likely engine edge case in
   another), so a blind "always keep bench≥2" rule is not obviously safe. If candidate 1 is gated and
   the deck-cut hypothesis confirmed, this may resolve on its own without a separate code change —
   re-evaluate after candidate 1's gate results before investing here.
4. **(low priority follow-up, not a v048 target)** The degenerate short/stalled-game pattern found in
   episode 86933558 (69 steps, turn stuck at 4, repeated near-identical prompts, abrupt unresolved end)
   — worth a narrow check across a handful more short (<80-step) losses to see if it recurs before
   deciding whether it's an engine issue worth reporting, but it's not an agent-logic lever.

## Data files

- `data/episodes/v044_v046_forensic.csv` — 129 rows, one per episode.
- `data/episodes/v044_v046_forensic_detail.json` — per-turn timelines, 45 flagged episodes.
- `data/episodes/v044_v046_other_deck_clusters.json` — deck-signature clusters for `archetype_truth=="other"`.
- `data/episodes/v044_v046_forensic_summary.md` — full win-vs-loss cross-tabs (source for all tables above).
