# v070_grimmsnarl_hybrid — design spec

Re-decked copy of a public Kaggle notebook's attention-BC pilot, run with **our** deck instead
of theirs. This is a "swap the deck, keep the brain" submission — no training, no pilot code
changes.

## Source

Public Kaggle notebook: `prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1`
(downloaded into `submissions/external_prvsiyan_grimmsnarl_hybrid/`, see memory
`project_prvsiyan_grimmsnarl_replay_hybrid_competitor`). An attention-based behavioral-cloning
network (`bc_agent.py`, torch, CPU-only) trained by imitation on 125 completed public replays of
an "Rmy" Grimmsnarl/Froslass deck — a decklist ~naunju-style, essentially the same archetype as
our own Grimmsnarl/Munkidori line (jiatu.l-derived), differing from our current best deck
(`submissions/v064_starmie_scaling/deck.csv`) by **exactly one card**: their 1137 (Tool Scrapper,
Item) vs. our 1197 (Xerosic's Machinations, Supporter — hand disruption).

`results/gates/grimmsnarl_hybrid_vs_v064_comparison.md` (2026-07-23, N=1000 per cell,
cg_native) found this external hybrid pilot **beats our v064_starmie_scaling pilot outright**
on their own deck: direct h2h hybrid-vs-v064 = 0.615 WR (v064-vs-hybrid = 0.398, consistent
across independent seeds), and hybrid also out-scores v064 against nearly every shared pool
opponent (v009a_crustle, v044_archaludon_abomasnow, v049_bench_sticky_guard, both external
Alakazam variants, both external Lucario variants) — only the Alakazam mirror is roughly a
coin-flip. That result is the sole justification for this submission: if the external pilot is
simply a stronger policy on (almost) our own decklist, swapping our deck onto their checkpoint
should be a strict win over shipping v064 again, provided the 1-card deck delta doesn't matter
to the model (see below).

## What changed vs. `submissions/external_prvsiyan_grimmsnarl_hybrid/`

- **`deck.csv`** — replaced with `submissions/v064_starmie_scaling/deck.csv` (our deck: card
  1197 Xerosic's Machinations instead of their 1137 Tool Scrapper; all other 59 cards identical).
- **`main.py`** — external `main.py`'s control flow (base BC prediction + two deterministic
  replay-validated overrides, `_punk_up_override`/`_lowest_hp_override`) is preserved verbatim.
  Added: a decision-logging wrapper (`DECISION_LOG`, stderr `[dec] ...` lines — turn, select
  context, option count, chosen index, which decision path fired
  `bc_model`/`override:punk_up`/`override:lowest_hp`, and the model's own per-option logits
  (`score_values`) plus the `no_action_value` threshold as a score/margin analogue, since there
  is no human-readable heuristic "reason" string for a neural policy). Deck-select calls
  (`obs.select is None`) are not logged. The logging path is wrapped in `try/except` and never
  affects the returned action.

## What did NOT change

- **`bc_agent.py`** — byte-identical copy (verified `md5`) of the external file. Not edited: the
  checkpoint loader (`load_policy`) verifies a static `feature_hash` against `card_feat_table` at
  load time and refuses to load if the input encoding drifts, so touching this file risks
  breaking checkpoint compatibility for no benefit.
- **`model.pt`** — byte-identical copy (verified `md5`) of the external checkpoint. Not
  retrained. Deck composition is not baked into the weights: `bc_agent.predict_action` builds a
  `_DECK_COUNTER` from whatever `deck.csv` is present at runtime and folds deck-remaining
  features in dynamically (`_deck_remaining_features`), so the same checkpoint is expected to
  adapt correctly to our decklist without retraining. `hybrid_ourdeck_vs_hybrid_orig` in the gate
  (see below) is the empirical check of that assumption.
- **`cg/`** — copy of `src/cg` (gitignored engine; every submission dir needs its own copy, see
  memory `project_submission_cg_dir_gitignored`).

## Files

- `main.py` — external control flow + decision logging (see above).
- `bc_agent.py` — untouched (attention-BC model + encoder + `load_policy`/`predict_action`).
- `model.pt` — untouched checkpoint.
- `deck.csv` — our deck (= `v064_starmie_scaling/deck.csv`).
- `cg/` — game engine (copy of `src/cg`).

## Verification performed

1. **exec-smoke** (`scripts/v070_exec_smoke.py`, run inside `pokemon-tcg-sim-torch:latest`,
   linux/amd64 Docker): `main.py` loaded via `exec()` into a bare namespace with no `__file__`,
   staged at `/kaggle_simulations/agent/`, process `cwd` deliberately left at `/tmp` (not the
   staged dir) — the exact historical failure mode for "no `__file__`" bugs. 6 games vs.
   `v009a_crustle` (which itself imports `lightgbm`, the other half of the known
   torch+lightgbm OpenMP double-init crash — worked around with
   `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`), 836 total moves, 0 exceptions, decision log
   confirmed exercising `bc_model` and both override paths. PASS.
2. **Comparative gate** (`scripts/gate_v070_grimmsnarl_hybrid_vs_pool.py`, `cg_native`,
   macOS arm64, N=300 exploratory per pair): reuses `CORE_PAIRS` from
   `scripts/gate_v064_regression.py`, plus an explicit `v064_h2h` and a direct
   `hybrid_ourdeck_vs_hybrid_orig` (this submission vs. `external_prvsiyan_grimmsnarl_hybrid`,
   same checkpoint, deck-only difference) to isolate the effect of the 1-card deck swap.
   See `results/gates/v070_grimmsnarl_hybrid_gate_report.md` for numbers and the explicit
   N=300-is-exploratory-only disclaimer (memory `project_v065_built_gated`: a prior N=300 smoke
   read 52% h2h where the N=1000 full gate told a materially different story).

Not submitted to Kaggle by this change — local build + gate only.
