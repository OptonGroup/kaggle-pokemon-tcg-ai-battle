# v061 feasibility verdict — deterministic pilot for the jiatu.l Grimmsnarl/Munkidori deck

**Date:** 2026-07-21 · **Branch:** worktree-v061-grimmsnarl-heuristic · **NOT submitted** (project rule: upload only on explicit user "ок"; v050 is in the live window).

## Question
Track B (the μ1130 jiatu.l Grimmsnarl/Munkidori deck) hit a BC-ranker ceiling: v060's LightGBM
ranker was bimodal (Alakazam 0.70–0.81, tanks 0.04–0.19; offline top-1 0.541). Our best LIVE
agents (v034 μ865.7 / v036 μ827.3) are deterministic matchup-heuristics, not RL/BC. **Can that
v034/v036 method be ported onto this deck?**

## Verdict: **FEASIBLE — strongly positive.**
The deterministic pilot (guards on raw obs_dict, v036-style greedy `score_option → choose_options`
on typed obs, LightGBM removed) **beats v060 head-to-head 0.822** on the same deck and is
Wilson-significantly better on 4 of 5 measured opponents, parity on the 5th.

## Local gate (cg_native, N=300 seat-balanced, results/gates/v061_core_gate.json)

| opponent | v061 | v060 baseline | Δ | note |
|---|---|---|---|---|
| v017 Alakazam | **0.937** [0.903,0.959] | 0.700 | **+0.237** | our best matchup, hugely improved |
| v016 Alakazam | **0.947** [0.915,0.967] | 0.800 | **+0.147** | |
| v044 Archaludon ex | **0.193** [0.153,0.242] | 0.064 | **+0.129** | tank cohort ~3× (plan goal) |
| v049 bench-sticky | **0.190** [0.150,0.238] | 0.066 | **+0.124** | tank cohort ~3× |
| v009a Crustle | 0.168 [0.130,0.215] | 0.189 | −0.021 | CI overlap = parity (Rock-Inn wall = deck's mechanical limit) |

**v061 vs v060 direct mirror (same deck, method-vs-method): 0.822** (246W/53L/1D, CI [0.774,0.861]),
results/gates/v061_vs_v060_mirror.json. 0 agent exceptions in every run.

Checkpoint-2 criterion ("Wilson-significant win on the tank cohort while preserving Alakazam
0.70–0.81") is **met and exceeded** — Alakazam went to 0.94–0.95, tanks tripled.

## Offline legibility (89,483 expert-won decisions, results/gates/v061_legibility.json)

- **MAIN-only top-1 = 0.499** (BC ranker baseline 0.541). Below the checkpoint-1 ≥0.55 line on
  the strict index metric — **but the gate proved this DID NOT translate to worse play** (the
  opposite: +0.24 vs the same Alakazam opponent). The offline shortfall was traced, not waved away:
  - Dominant bucket **adrena_trigger (14.6% of decisions, func 0.42)**: measured directly — when a
    Munkidori Adrena-Brain ability is available, the expert fires it in ~50% of *sub-decisions*
    **at any opponent HP** (no board threshold separates fire/no-fire; fire-rate ≈0.43–0.62 flat,
    0.72 vs tanks). Adrena-Brain is *once-per-turn*: the expert fires it once per turn but at a
    variable point within the turn; our guard fires it at the first opportunity. On **isolated
    pairs** (which can't see the ability was already spent that turn) this reads as "over-firing,"
    but in a real game both fire it once per turn — a per-decision-metric artifact of a
    once-per-turn ability, aggregate-equivalent. The gate confirms it costs nothing.
  - `energy_attach` / `attack_choice`: within-turn action *ordering* (we develop-then-attack, the
    expert interleaves) — same multiset of actions per turn, identical outcome.
  - ~6/60 cards differ from the expert's exact list (e.g. Handheld Fan 1161) → unmatchable noise.
- Guard-mechanical contexts reproduce cleanly: activate 0.94, skill_order 1.00, draw_count 1.00,
  discard_energy 0.98, setup func 1.00, damage_splash 0.81.

**Interpretation:** strict per-decision top-1 systematically *under*-reports a deterministic
priority-scorer's fidelity (it can't model once-per-turn timing or within-turn ordering). The
gate/live is the arbiter (confirmed ≥4× in project history), and the gate is decisive.

## What was built / fixed (see submissions/v061_grimmsnarl_heuristic/DESIGN.md)
- Skeleton ported from v036; LightGBM/features/pandas/EN_Card_Data.csv **removed** (3 fewer
  live-failure surfaces; bundle has no numpy/pandas/lightgbm).
- Guard edits: (a) Froslass 104 re-tiered to "tanky utility" (Frost Smash {W} is uncastable — 0 {W}
  in deck; v060 mis-tiered it as an attacker); (b) Adrena-Brain opportunistic trigger widened to
  every matchup (correct direction — expert fires it broadly, not only vs tanks); (c) `_ko_guard`
  moved off pandas/CSV onto cg.api `CARD_DB`/`ALL_ATTACKS`.
- **Legibility-driven fixes** (revisit-before-gate loop): `_ko_guard` now only front-runs the
  scorer for a **game-winning** lethal (a non-winning KO is sequenced last by the scorer, matching
  the expert's develop-first/attack-last — lifted forced_lethal from 0.20); `attach_target_score`
  now keeps every attach **strictly positive** (energy on board > energy stranded in hand — spend
  the drop every turn, never END-over-attach).

## Live-safety (mandatory pre-submit, exec-smoke, scripts/v061_exec_smoke.py)
Ran in Docker linux/amd64 (pokemon-tcg-sim:latest), agent exec()-loaded WITHOUT `__file__` from
/kaggle_simulations/agent: **PASS** — scorer_ok=True, damage_table_ok=True, 0 exceptions over 6
games, scorer path exercised (scored=423, trivial_fallback=4, internal_exc=0).

## Corrections to project memory (verified against the engine)
- **Munkidori 112 in FRAGILE_SUPPORT_IDS is CORRECT, not a bug** (memory had flagged it): Mind Bend
  costs {P}, deck runs 0 {P} → uncastable. Only 648/647/646 can attack.
- **Froslass 104 was genuinely mis-tiered in v060** (as an attacker): Frost Smash costs {W}, 0 {W}
  in deck. Froslass is a pure passive engine (Freezing Shroud feeds Adrena-Brain each Checkup).

## Recommendation
Feasibility is proven locally. Next step is a **user decision**: the local gate is only a
regression filter — live is the sole arbiter (v012 precedent: 0.70 local gate → live 499). v061 is
a candidate worth a live probe when a slot opens, but v050 is currently in the live window and the
project rule forbids uploading without explicit "ок". No submission was made.
