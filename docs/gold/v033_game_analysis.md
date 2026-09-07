# v033_plan_d_ppo (pure-NN PPO, no MCTS) — live replay analysis

50 games, submission 54779509 (`v033_resub`). Data:
`data/episodes/episode-<id>-replay.json` + `episode-<id>-agent-{0,1}-logs.json`,
parsed by `scripts/analyze_v033.py`, per-game detail in
`data/episodes/v033_analysis.json`. Runtime forensic by
`scripts/runtime_forensic_v033.py`.

## Pre-flight (A0)

- sha256 of `submissions/v033_plan_d_ppo/plan_d_ppo_v2_best.pth` and
  `models/plan_d_ppo_v2_best.pth` match byte-for-byte
  (`d230729663f14087104702cd21800c8987b988b707e609788052eeda548af483`).
- Local god-mode-free gate (`scripts/gate_expanded.py`, native macOS engine, both
  sides via `cg_obs`) reproduced: n=30 fresh games, WR=76.7% [59.1%, 88.2%]
  (existing `results/gates/v033_gate_expanded.json` n=300 record: WR=79.2%
  [74.2%, 83.4%]) — consistent, no regression, sanity check passes.
- Exec-smoke: `[v033] PPO model loaded OK` printed, `nn_decisions=988`,
  `fallback=0`, `exceptions=0` over the 30-game sanity run. (The gate script's
  own `model_loaded=False` print is a false alarm — it samples `_model` at
  import time, before the lazy singleton's first call; the model does load on
  the very first decision, as `nn_decisions>0` and the stderr line confirm.)

## Runtime forensic (A2) — STOP-CRANE did NOT trigger

All 50 live episodes' agent stderr logs contain **`[v033] PPO model loaded OK`**
and **zero** occurrences of any known silent-fallback/crash signature
(`import failed`, `model load failed`, `nn_action failed`, `Traceback`,
`No module named`, `numpy.dtype size changed`, `NameError`, `ImportError`).
The NN genuinely ran in all 50 live games — this is not a fallback-agent
result masquerading as v033.

One infra note found during download: Kaggle only grants log access to
**your own** agent slot, not slot 0 unconditionally — in 23/50 episodes our
submission was agent index 1, and `kaggle competitions logs <id> 0` 403'd
while `... 1` succeeded. `scripts/fetch_v036_episodes.sh` (as written, always
requesting index 0) silently drops these logs with no error surfaced by its
429-only check; they had to be re-fetched manually with the correct index
derived from `replay.json`'s `info.Agents` list. Worth patching the fetch
script if reused again.

## Score

**W-L-D = 25-25-0, winrate 50.0%** (50/50 games downloaded and parsed).

This is far below the local god-mode-free gate's 79% (and even the fresh 30-game
gate reproduction's 77%) — see "God-mode gap" below.

## Winrate by opponent archetype

| Archetype | Games | W | L | WR |
|---|---|---|---|---|
| Mega Lucario ex | 12 | 8 | 4 | 67% |
| Starmie | 9 | 4 | 5 | 44% |
| Alakazam | 8 | 1 | 7 | 12% |
| Crustle/Dwebble (mirror) | 6 | 2 | 4 | 33% |
| Grimmsnarl ex/Munkidori | 3 | 3 | 0 | 100% |
| Archaludon | 3 | 2 | 1 | 67% |
| Dragapult ex | 3 | 3 | 0 | 100% |
| Grimmsnarl ex (no Munkidori seen) | 1 | 1 | 0 | 100% |
| Unknown/rogue (4 distinct lineups: Team Rocket Articuno/Mewtwo, Applin/Celebi/Dipplin, Budew/Dudunsparce, Iono's Bellibolt/Kilowattrel) | 4 | 1 | 3 | 25% |
| Unknown (Larvitar/Pupitar/Tyranitar) | 1 | 1 | 0 | 100% |

**Known-meta archetypes combined (excl. Unknown buckets): 45 games, 23-22, WR 51.1%.**
**Alakazam alone: 8 games, 1-7, WR 12%** — by far the single worst matchup, and
the dominant loss driver (see below).

## Top loss causes (25 losses)

1. **Alakazam locks us out of attacking entirely — 7/8 Alakazam losses (28% of
   all losses) never landed a single attack the whole game** (episodes
   86486137, 86485035, 86483949, 86481768, 86480129, 86470539, 86467073 — all
   `atk=0`). Games end fast (turns 3-15, median ~6). This is the same
   structural Crustle-vs-Alakazam weakness seen across *every* pilot tried in
   this project regardless of architecture (v012 iono-heuristic live 499.2,
   v017_alakazam_heur WR 27% N=11) — strong evidence this is a deck-archetype
   problem, not a PPO/god-mode-specific policy bug.
2. **Starmie and Crustle-mirror underperform their god-mode/local-gate
   expectations** — Starmie 44% (4-5), mirror 33% (2-4). Combined with
   Alakazam, these three archetypes alone account for 16/25 losses (64%).
   Mega Lucario ex (our best matchup, 67%) and the fast-aggro decks
   (Dragapult ex 100%, Grimmsnarl 100%) are fine — the losses concentrate in
   exactly the match-ups that call for longer-horizon, disruption-aware play
   (mirror, Starmie's Psychic-type answers, Alakazam's burst) where a
   god-mode-trained policy would be expected to degrade most if it leans on
   information it can no longer see.
3. **Never developed a second Pokemon (bench stayed at 0 all game) — 6 games
   total, 5 of them losses** (86488329, 86481768, 86480129, 86470539,
   86463554 — losses; 86483413 is the one exception, a win). Same
   mono-attacker pattern as v024; not fatal on its own (won once against a
   weak rogue lineup) but fatal vs anything that needs card advantage or a
   second attacker — 3 of these 5 losses are also Alakazam games, compounding
   with loss cause #1.
4. **Hand clog (bench empty + hand ≥8 by turn 5) — 9/50 games total**, mixed
   W/L (not a clean loss signal on its own, consistent with v024's finding
   that this tracks the energy-heavy 60-card list generally, not a policy
   defect).
5. A **parsing caveat**: spot-checking raw action indices against option lists
   in multi-sub-decision turns (e.g. episode 86483949 turn 4) found at least
   one case where the logged `action` index falls outside the `option` list
   shown in that same view (index 6 into a 2-option list) — an artifact of how
   the replay JSON compresses multi-step turns, inherited unchanged from the
   v024/v036 analyzer lineage. This means the `declined_attack_turns` flag
   (13 games flagged) should be read as directional, not turn-exact.

## Why we win

- Mega Lucario ex (8-4, 67%), Dragapult ex (3-0, 100%), and Grimmsnarl
  ex/Munkidori (3-0 / 4-0 combined, 100%) are comfortably positive — these are
  fast/aggressive archetypes where Crustle's straightforward beatdown plan
  wins races.
- **Retreat is actually used** (unlike v024's pure-transformer policy, which
  retreated 0/94 legal opportunities): v033 retreated in 7 different games
  (`cnt_retreat` 1-2), so this specific action head is not dead in the PPO
  checkpoint.
- No retreat-oscillation loops, no stuck-on-Dwebble-forever cases, and attack
  was available at least once in every single game (never fully locked out of
  the option, even in the Alakazam blowouts — it just didn't arrive fast
  enough before losing).

## God-mode gap (A4)

Local god-mode-free gate WR ≈ 77-79% (both fresh 30-game reproduction and the
existing 300-game record) vs **live WR 50.0%** — a **~28-29 percentage-point
gap**. Per the plan's interpretation scale (WR≪0.65 ⇒ gap real and strong),
this lands solidly in "real and strong."

Caveat on attribution: the local gate's opponent is always `v009a_crustle`
(a single fixed IL/LightGBM pilot playing a single deck), while live opponents
are the actual matchmaking pool (12 different Mega Lucario players, 9
different Starmie players, 8 different Alakazam players, etc., each with their
own skill level and deck-list variance). So part of the 28pp gap is a
**generalization gap** (one god-mode training opponent → a whole live pool)
independent of, and compounding with, the god-mode/hidden-hand mismatch
described in main.py's own CAVEAT docstring. The data here cannot cleanly
separate the two effects; both plausibly contribute, and this analysis alone
does not prove the hidden-hand mechanism specifically (no smoking-gun decision
was found where the policy's choice only makes sense if it could see the
opponent's hand — see below). What it does establish is that the god-mode
local gate number (~79%) is **not a usable live-performance predictor** for
this checkpoint, regardless of which mechanism dominates.

Qualitative deep-dive of losses (86470539, 86483949, 86486504, 86478470,
86463554 examined at the decision level): no clear-cut example surfaced of a
decision that would only make sense with opponent-hand visibility (e.g., an
inexplicable early retreat/pass that "predicts" an unseen threat). The
dominant loss pattern instead reads as **tempo loss** — Alakazam/Starmie/mirror
opponents completing their setup or disruption plan before Crustle gets a
single attack in — which is consistent with a generically under-strength
policy in these specific matchups (whether due to god-mode training gap,
single-opponent overfitting, or both) rather than a distinctive
"plays-around-invisible-cards" signature.

One side observation, not part of the god-mode question but notable: v033
went first in only 5/50 games (10%) — 3W-2L when first vs 22W-23L when
second. Too small a split to draw a first/second-player conclusion from, and
turn order is matchmaking-determined, not a policy choice.

## Bottom line

- Runtime is clean: NN ran with 0 exceptions in all 50 live games — the 50%
  live WR is a genuine measurement of the PPO policy's live performance, not
  an artifact of silent fallback.
- The god-mode gap is real and large (~29pp), but the evidence collected here
  supports it as *a* cause among plausibly several (single-opponent
  overfitting in the local gate, live pool diversity, the documented
  hidden-hand training/inference mismatch) rather than proving hidden-hand
  mismatch in isolation.
- Alakazam is the single clearest, most actionable finding: 12% WR, 7/8
  losses with **zero attacks landed the entire game** — a known
  Crustle-archetype weakness reproduced yet again under a completely
  different (PPO/NN) pilot, reinforcing that this is a deck-level problem
  rather than something a policy patch alone would fix.

Sample caveat: 50 games is a moderate sample (above the plan's 15-40
"partial" tier, just under its 60-game "full verdict" tier); per-archetype
breakdowns (n=1 for two "Unknown" buckets, n=3 for Dragapult/Grimmsnarl) are
too small individually to be conclusive, though the aggregate 50% WR and the
Alakazam-specific pattern (n=8, 7 of which are the same zero-attack failure
mode) are both large enough within-sample to be a real signal rather than
noise.
