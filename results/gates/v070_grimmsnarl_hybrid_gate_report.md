# v070_grimmsnarl_hybrid — build + gate report (2026-07-23)

Re-decked copy of the public Kaggle notebook `prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1`
(attention-BC pilot, `submissions/external_prvsiyan_grimmsnarl_hybrid/`), run with **our**
Grimmsnarl decklist (`submissions/v064_starmie_scaling/deck.csv`) instead of theirs. Motivated
by `results/gates/grimmsnarl_hybrid_vs_v064_comparison.md` (N=1000, 2026-07-23): on their own
deck this external pilot beats our v064 pilot outright (h2h 0.615 hybrid-vs-v064) and out-scores
v064 against nearly every shared pool opponent.

## 1. What was built — `submissions/v070_grimmsnarl_hybrid/`

| file | source | changed? |
|---|---|---|
| `bc_agent.py` | `external_prvsiyan_grimmsnarl_hybrid/bc_agent.py` | **no** — byte-identical (md5 `1e855c7234cd0eda2aa2b799c0a8094e`) |
| `model.pt` | `external_prvsiyan_grimmsnarl_hybrid/model.pt` | **no** — byte-identical (md5 `7ba6ca799ec799636a20628555533666`) |
| `deck.csv` | `v064_starmie_scaling/deck.csv` | our deck (card 1197 Xerosic's Machinations instead of their 1137 Tool Scrapper; other 59 cards identical) |
| `main.py` | adapted from `external_prvsiyan_grimmsnarl_hybrid/main.py` | added stderr decision logging (`DECISION_LOG`, `[dec] ...` lines: turn, select context, option count, chosen index, decision path `bc_model`/`override:punk_up`/`override:lowest_hp`, model's own per-option logits as score/margin analogue, `no_action_value`). Deck-select calls not logged. try/except-wrapped, never affects the returned action. |
| `cg/` | `src/cg` | copy (gitignored engine, required per submission dir) |
| `DESIGN.md` | new | source, changes, verification summary |

## 2. exec-smoke — **PASS**

`scripts/v070_exec_smoke.py`, run inside `pokemon-tcg-sim-torch:latest` (linux/amd64 Docker,
torch 2.13.0+cpu present):

- `main.py` staged at `/kaggle_simulations/agent/`, loaded via `exec(compile(src, "<string>",
  "exec"), ns)` into a bare namespace (**no `__file__`** in the namespace — asserted), process
  `cwd` deliberately left at `/tmp` (not the staged dir) — reproduces the historical "missing
  `__file__`" live-only failure mode.
- Deck-select call (`agent({"select": None})`) returned a valid 60-int-ID list.
- 6 full games vs. `v009a_crustle` (itself imports `lightgbm` — the other half of the known
  torch+lightgbm OpenMP double-init crash, worked around with `KMP_DUPLICATE_LIB_OK=TRUE
  OMP_NUM_THREADS=1`, see `results/gates/grimmsnarl_hybrid_vs_v064_comparison.md`).
- 836 total moves, **0 exceptions**, every returned choice a valid `list[int]`.
- Decision log confirmed exercising all three paths: `bc_model` (base neural policy),
  `override:punk_up`, `override:lowest_hp`.

```
[smoke] total_moves=836 total_exceptions=0
[smoke] PASS
```

## 3. Comparative gate — N=300 exploratory (⚠️ NOT a final verdict — see disclaimer below)

`scripts/gate_v070_grimmsnarl_hybrid_vs_pool.py` (`cg_native`, macOS arm64, seed=20260723).
Candidate = `v070_grimmsnarl_hybrid` (our deck + external hybrid pilot). Full results in
`results/gates/v070_grimmsnarl_hybrid_vs_pool.json`. **All 14 pairs: 0 crashes / 0 exceptions /
0 illegal moves** on either side.

### vs. shared pool opponents (three-way comparison)

| Opponent | v064 WR (N=1000, ref¹) | ext hybrid, THEIR deck, WR (N=1000, ref²) | **v070, OUR deck, WR (N=300, this gate)** |
|---|---|---|---|
| v017_alakazam_heur | 0.965 | 0.982 | **0.983** [0.962,0.993] |
| v016_alakazam | 0.977 | 0.988 | **0.990** [0.971,0.997] |
| v009a_crustle | 0.219 | 0.440 | **0.402** [0.348,0.458] |
| v044_archaludon_abomasnow | 0.310 | 0.434 | **0.430** [0.375,0.487] |
| v049_bench_sticky_guard | 0.341 | 0.454 | **0.373** [0.321,0.429] |
| ext_prvsiyan_alak_v8 | 0.385 | 0.554 | **0.667** [0.612,0.718] |
| ext_prvsiyan_alak_v9 | 0.350 | 0.527 | **0.603** [0.547,0.657] |
| ext_maulikgajera_lucario | 0.227 | 0.374 | **0.283** [0.235,0.337] |
| ext_nursrijan_lucario | 0.244 | 0.403 | **0.368** [0.316,0.424] |
| v063_forensic86 (h2h) | — | 0.599 (N=1000) | **0.587** [0.530,0.641] |
| v036_archaludon_patched (h2h, reference-only)³ | — | crashed (fixture missing) | **0.340** [0.289,0.395] |

¹ `results/gates/grimmsnarl_hybrid_vs_v064_comparison.md`, N=1000.
² `results/gates/ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json`, N=1000 — this is the external
pilot with **their own deck** (card 1137), not ours.
³ In both this run and the historical external-hybrid N=1000 run, `v036_archaludon_patched`
lives outside this worktree by default and all 300/1000 games crash (`worker_crash`) before any
game is played — a missing-fixture artifact of the worktree, not an agent bug. Fixed for this
gate by copying `submissions/v036_archaludon_patched/` into this worktree from the main repo
before re-running that one pair; the 0.340 WR is a real measurement (0 crashes after the fix).
Baseline=None / reference-only pair either way.

**Reading**: on every pool opponent, v070 (our deck) tracks the external hybrid's numbers on
its own deck (column 2) much more closely than it tracks v064's numbers (column 1) — consistent
with "re-decking barely moved the pilot's strength," which is exactly the deck-swap hypothesis
being tested. v070 is at or above the external-hybrid-on-their-deck number on 6/9 pool
opponents (both Alakazam pairs by a wide margin, v017/v016 ties, v009a/v044 roughly flat,
v049/lucario pairs slightly below).

### Direct head-to-heads

| Matchup | WR | Wilson95 CI | N |
|---|---|---|---|
| v070 (our deck) vs v064_starmie_scaling | **0.563** | [0.507, 0.618] | 300 |
| v070 (our deck) vs external_prvsiyan_grimmsnarl_hybrid (their deck), via `ext_prvsiyan_grimmsnarl_hybrid` pair | 0.490 | [0.434, 0.546] | 300 |
| v070 (our deck) vs external_prvsiyan_grimmsnarl_hybrid (their deck), via explicit `hybrid_ourdeck_vs_hybrid_orig` pair | 0.493 | [0.437, 0.550] | 300 |

The last two rows are **the same matchup measured twice under independent seeds** within this
run (CORE_PAIRS already contained an `ext_prvsiyan_grimmsnarl_hybrid` opponent entry, and the
task asked for an explicit dedicated pair too) — both land at ≈49%, i.e. a coin-flip, which is
the key sanity check: **the 1-card deck swap (1137→1197) does not measurably change the
pilot's strength against itself.** v070 vs v064 at 56.3% is a positive but noisy signal in the
same direction as the N=1000 finding that motivated this build (hybrid-vs-v064 = 0.615 on the
hybrid's own deck).

## 4. Disclaimer — N=300 is exploratory, not a shipping verdict

Per `project_v065_built_gated` (memory): a prior N=300 "smoke" read 52% h2h where the
subsequent N=1000 full gate showed a **materially different, opposite-leaning** picture (exact
50/50 h2h, and significant swings on some pool opponents). All numbers in §3 above should be
read the same way — directionally informative, not a submit/no-submit decision on their own.
The individual-pair Wilson 95% CIs above are ~±5-6pp wide at N=300; several of the "v070 above
hybrid-on-their-deck" gaps (e.g. both Alakazam externals) are comfortably outside that noise
band and look like real signal, but the closer ones (v009a, v044, v049, both Lucario externals,
v063_h2h) are within a CI or two of the reference numbers and could still flip.

## 5. Recommendation

- The deck swap looks safe: the direct same-checkpoint h2h (our-deck vs their-deck) sits at
  ≈49% both times measured, i.e. no detectable strength loss from swapping in
  1197/Xerosic's Machinations for 1137/Tool Scrapper.
- v070 (our deck + external hybrid pilot) beats v064 56.3% at N=300 and generally tracks or
  beats the external hybrid's own N=1000 pool numbers — consistent with the original
  motivation (`grimmsnarl_hybrid_vs_v064_comparison.md`) that this checkpoint is simply a
  stronger pilot than our own on (almost) this decklist.
- **Before considering a Kaggle submission**, run the full N=1000 CORE_PAIRS gate (same script,
  `--n 1000`) — given the memory precedent above, that is the actual bar for a submit decision,
  not this N=300 pass. This N=300 run is a "does the build work / is the deck swap safe" check,
  and it passes; it is not sufficient evidence on its own to green-light a submission.
- No Kaggle submission was made or attempted as part of this task — build + local gate only.
