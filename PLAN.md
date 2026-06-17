# PLAN — Pokemon TCG AI Battle (execution planner)

> **This file is the single source of truth for execution.** A junior model reads it,
> finds the first unchecked task, does it, then updates the **Status Log** at the bottom.
> Read `CLAUDE.md` first for project facts; this file is the *what to do next*.

Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
Sim deadline: **2026-08-16** → ~2 weeks extra games → final ~2026-09-01.
Current standing (2026-06-17 ~10:00): **CHAMPION = v002_lucario μ=798.8, RANK 138/570** (top ~24%). v002 record on
ladder = ~7W-5L (12 public games, last at 03:58 — μ STALLED, not climbing). v001_random μ=517.4, retired. v003_search
FAILED the gate (~50% vs v002) — shelved. **LB INFLATED hard in 1 day** (field now 570 teams): rank1=1160.8, top20≈1005,
top50≈908, top100≈841, me(138)=798.8, rank200≈740. We are NOT near top-20 — the whole field's μ climbs as they play,
so **sitting still = sliding down relatively.** NOTE: μ is LIVE; re-read via CLI (`leaderboard -d`, `episodes <subid>`).

---

## 0. How a session works (read every time)

1. Open this file. Find the **NEXT ACTION** pointer (just below) and the first unchecked `[ ]` task.
2. Do exactly that task. If it's heavy/mechanical, spawn a **lighter subagent** for it.
3. Before any submission, run the **Submission Gate** (§Guardrails). No gate pass → no submit.
4. Append a dated entry to the **Status Log** (bottom). Update checkboxes. Update `results/submissions.csv`.
5. If you discover a fact worth keeping across sessions, write it to memory (see CLAUDE.md memory rules).

**NEXT ACTION:** Phase 3 (deck gauntlet) — human chose this over IL/search 2026-06-17. KEY CONSTRAINT discovered:
the pilot (`src/agents/heuristic_agent.py` = v002 main.py) is **HARD-CODED to Mega Lucario ex** (card IDs 673–678,
Mega Brave attack 983, fighting-weakness ×2, etc.). So the gauntlet CANNOT just "pilot any deck with our best agent" —
our best agent only plays Lucario. Realistic Phase-3 win = **tune the Lucario list with the same pilot** (Source A):
Lucario is already the best sample deck (60.4%) and the pilot is built for it. CAVEAT: with a frozen deck-specific
pilot the deck-tuning gain ceiling is modest and HARD to prove over gate noise (±5.7% on 300 games) — same
"indistinguishable from champion" risk that shelved v003. Plan: build `src/eval/gauntlet.py` (round-robin variants,
same pilot) → author pilot-SAFE Lucario variants (count tweaks only: energy 13±1, Switch/Boss/Cape copies; NO new
complex-cost cards the pilot can't handle) → smoke → gate winner vs v002 (300g) → submit only on PASS + human "ок".
Bigger ceiling (net-deck Abomasnow archetype that reached μ=1030) needs a NEW deck-specific pilot = much larger effort.

---

## Guardrails (NON-NEGOTIABLE)

- **Submission Gate.** Only submit a candidate if ALL hold:
  1. It beats the current **champion** in local self-play: winrate **≥ 55%** over **≥ 300 games**
     (≥150 going first, ≥150 second), AND Wilson 95% lower bound **> 50%**.
     (Run big gauntlets on Kaggle notebook for speed; Docker for quick smoke checks.)
  2. It passes Kaggle **self-validation** (the validation episode on upload completes without error).
  3. We are **not** in the deadline freeze (see below).
- **Champion bookkeeping.** `results/champion.txt` holds the current best version dir name.
  We must ALWAYS be able to re-submit the champion unchanged. Never delete a champion's dir.
- **Daily budget.** ≤ **5 submissions/day**. Reserve ≥1 slot for a champion rollback. Never burn all 5 on experiments.
- **Final-2 rule.** Only the **latest 2** submissions count for the final. So the last two uploads must be
  our **champion** + at most one well-tested variant. Never let an experimental/unvetted agent be one of the last two.
- **Deadline freeze.** From **2026-08-11** (5 days before sim deadline): no experimental submissions.
  Ensure the last two submissions are the champion (and the single best vetted variant). Stability > marginal gains.
- **Ladder rollback.** After a submit, watch episodes. If ladder μ trends **below** the champion after ≥10 public
  episodes, re-submit the champion (roll back) and investigate.
- **Engine is read-only.** Never edit files under any `cg/` dir. Copy `src/cg/` into each new `submissions/vNNN/` dir.
- **Format.** `.tar.gz` (NOT zip) via `tar -czvf submission.tar.gz *` from inside the submission dir.
  `main.py` at top level, plus `deck.csv` and `cg/`. Use `./scripts/submit.sh <version> [msg]`.
- **Deck legality.** Max 4 copies of any card by name (basic energy unlimited), exactly 60 cards, ≥1 Basic Pokémon.
  **Ace Spec = max 1 copy per deck** (engine rejects 2+ with `errorType=4`; confirmed 2026-06-17 with Hero's Cape id=1159).
  Validate every candidate deck before use.
- **Time bank.** Each agent has a ~600s bank for the WHOLE game (not per move). Budget search so the bank lasts
  every decision in a long game; keep a safety margin. Going over the bank = TIMEOUT = instant loss.
- **Never throw.** Any unhandled exception or illegal selection = instant loss. Wrap the agent in a try/except that
  falls back to a always-legal move (e.g. first `maxCount` indices) so a bug degrades gracefully instead of losing.

---

## Facts established 2026-06-16 (env = `cabt` in kaggle-environments 1.14.10)

> Kaggle discussions/pages ARE accessible via the CLI (no WebFetch needed):
> `kaggle competitions topics list|show`, `kaggle competitions topic-messages <comp> <topic_id> -s top -n -1`,
> `kaggle competitions pages <comp> --content --page-name "<Name>"`. Env config lives in GitHub
> `Kaggle/kaggle-environments` → `kaggle_environments/envs/cabt/` (`cabt.json`, `cabt.py`).

- **Q-TIME → ANSWERED.** `cabt.json`: `actTimeout=0`, `runTimeout=3000`, `episodeSteps=10000`,
  `observation.remainingOverageTime=600`. Effectively each agent has a **~600s time BANK for the WHOLE game**
  (not per move). Deep search is viable — just budget so the bank lasts all decisions in a long game.
- **Q-LEGAL → ANSWERED** (verified in P0-1). Two rejection modes: (1) wrong length (≠60) → the Python wrapper
  `battle_start` raises `ValueError` before the C lib; (2) illegal composition (e.g. no Basic Pokémon) → `battle_start`
  returns `(None, start_data)` with `start_data.errorPlayer>=0` and `errorType` set (saw `errorType=3` = deck error).
  An invalid deck = INVALID status = loss. Always validate decks before submitting; callers must check `obs_dict is None`.
- **Q-RULES → ANSWERED** (thread 708586). Differences from official TCG are **minor and don't affect strategy**:
  (1) some unresolvable attacks are simply non-selectable; (2) Mega Zygarde ex "Nullifying Zero" coin order is
  auto left-to-right; (3) simultaneous-KO prize order differs but if both reach 0 prizes it's a **draw**.
  **Own decks ARE allowed** — build from the Data-tab card pool. Assume standard TCG rules otherwise (6 prizes).
- **CRASH/TIMEOUT/INVALID = INSTANT LOSS.** `cabt.py` gives the offending agent `reward=-1` and the opponent the
  win on any exception, timeout, or illegal selection. → The agent must NEVER throw and must stay within the bank.
- **Reward mapping:** `rewards[i]` pairs with `info.Agents[i].Name`. Don't assume we're player 0 — match by name.

### Still open
- **Q-DEPS → ANSWERED 2026-06-17 (P0-3, free sources, NO probe submission burned).** Submission runtime =
  `python-simulations-cpu` Docker built FROM `gcr.io/kaggle-images/python:v163` (full Kaggle Python image, Colab base,
  Python 3.12); the agent runs in-process (compile/exec) so the whole image package set is importable, NO internet at
  eval. AVAILABLE (confident): **numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, torch (CPU-only `+cpu`)**.
  onnxruntime is NOT confirmed present (only `onnx` format lib) → avoid it; bundle a wheel only if truly needed.
  Evidence: kaggle-environments `docker/Dockerfile` BASE_IMAGE + googlecolab/backend-info `pip-freeze.txt` (has
  lightgbm 4.6, torch 2.x+cpu, sklearn, xgboost) + Kaggle/docker-python v163 release notes. → Phase 4 IL is UNBLOCKED:
  use a LightGBM ranker or a small CPU torch MLP (numpy-only fallback is always safe).
- **Q-SEARCH → ANSWERED** (P2 probe, `src/eval/search_probe.py` in Docker, 2026-06-17): one `search_begin` ≈ 0.30 ms,
  one `search_step` ≈ 0.22 ms, a full rollout to terminal ≈ 88 steps ≈ 19.3 ms → **~52 rollouts/sec** under ARM
  emulation (faster native on Kaggle). With 50% of the 600s bank spread over ~120 decisions that's ~2.5 s/decision
  ≈ **~129 rollouts/decision** → e.g. K=10 determinizations × 12 options. **KEY:** the gate measured only **8 ms/move
  actual** → we are using a TINY fraction of the bank; search depth/K can grow ~30–50× before the bank is at risk.

## META & reference assets (from forum 708617 + sample notebooks)

- **There are 4 organizer sample decks** (each with a sample rule-based agent). ~15k self-play games:
  | Deck | Overall WR | vs Lucario | vs Dragapult | vs Iono | vs Abomasnow |
  |---|---|---|---|---|---|
  | **Mega Lucario ex** | **60.4%** | — | 50% | 78% | 52% |
  | **Dragapult ex** | **55.6%** | 50% | — | 64% | 53% |
  | Iono | 43.8% | 22% | 36% | — | 74% |
  | **Mega Abomasnow ex** (← our v001 deck) | **40.2%** | 48% | 47% | **26%** | — |
- **LADDER REALITY CHECK:** the #1 player (`platypus2000jp`, μ=1030) plays a **tuned Mega Abomasnow ex / Kyogre** —
  the SAME archetype as our v001. So the 4-deck winrates reflect *sample-agent* quality, not deck ceiling:
  **piloting + tuning dominate deck choice.** Don't discard Abomasnow — let the Phase-3 gauntlet (our best pilot) decide.
  Their revealed tech (ep 80204451): +Ultra Ball (1121), +Surfing Beach (1262), +Precious Trolley (1126) → net-deck lead.
- **Action:** ship **v002 = Mega Lucario ex** as a fast, competent baseline (forgiving deck + better sample agent),
  saved to `data/sample_agents/mega_lucario_ex/{main.py, deck.csv}`. But treat deck choice as OPEN until the gauntlet.
- **Net-deck access works:** leaderboard → `team-submissions <teamId>` → `episodes <subid>` → `replay <epid>`.
  Saved sample in `data/episodes/top_players/`. Replays reveal only partial decks → aggregate many to reconstruct.
- **IL data:** organizers will post a **daily export of top-rated episodes** in the forums (for BC/RL/IL) — watch for it.
- **Notebook workflow:** the cg engine is provided as a Kaggle dataset input (`cg-lib/cg`); the submission tar pulls
  `cg`, `deck.csv`, `main.py`. We already have `cg/` locally in `submissions/*/cg/`.

---

## Phase 0 — Infrastructure (P0) — build the loop before any strategy

> Goal: a junior model can write an agent, validate it, and gate-test it WITHOUT guessing.

- [x] **P0-1 Docker self-play harness.** ✅ DONE 2026-06-16. Created `docker/Dockerfile` (`--platform=linux/amd64`,
      py3.11, numpy/pandas), `.dockerignore`, `scripts/sim.sh`, `src/sim_harness.py` (`tournament()` → winrate +
      Wilson 95% CI + mean move time; `check_illegal_deck()`). Verified: `./scripts/sim.sh 20 42` ran in Docker →
      random-vs-random 12–8, winrate 0.600 CI [0.387,0.781], **mean move time 0.089 ms** (engine is cheap even
      emulated → search depth is bounded by OUR logic, not engine cost). Notes for future sessions:
      if `docker info` fails on macOS, run `open -a Docker` and wait; `random_agent` returns `[]` on deck-selection
      steps (caller already passed decks to `battle_start`); `battle_start` returns `(None, start_data)` on a bad
      deck — callers MUST check `obs_dict is None`.
- [ ] **P0-2 Timing/throughput probe on Kaggle.** Q-TIME already known (600s bank/game). Create
      `notebooks/selfplay.ipynb`: measure (a) games/min native, (b) wall-time per decision, (c) cost of one
      `search_begin`→`search_step` cycle and how many sims fit in the bank for a long game (→ Q-SEARCH).
      **Done when:** we know native throughput and a safe per-decision search budget.
- [x] **P0-3 Submission-runtime deps probe.** ✅ DONE 2026-06-17 (free sources, NO submission burned). See Q-DEPS
      above: runtime = full Kaggle Python image v163 (Colab base, py3.12), in-process exec → numpy/pandas/scipy/sklearn/
      **lightgbm**/xgboost/**torch(CPU)** all available; onnxruntime NOT confirmed (avoid). IL inference stack = clear.
- [x] **P0-4 Replay tooling.** ✅ DONE 2026-06-17. Harvested **189 top-player replays** (top-25 teams, μ 978–1160)
      into `data/episodes/top_players/` + our 15 → 204 games. `src/replays/parse.py` (stdlib-only, runs on macOS WITHOUT
      libcg — operates on raw replay JSON): `parse_replay()` → (obs_dict, chosen_option_indices, context, n_options, won,
      player) pairs; `revealed_cards()` → per-player card multiset. Stats: **19,440 (obs,action) pairs (7,721 from
      winners)**, MAIN=62.4%, avg 6.69 options/decision, ~21 distinct cards revealed/player/game. Replay step format:
      `steps[i][player]={action,observation,reward,...}`; the TRAINING pair is `(obs=steps[i].observation,
      chosen=steps[i+1].action)`; action = 0-based positional indices into the prior obs's `select.option`. (Other
      players' agent LOGS are 403 — only replays are downloadable. Organizer top-episode export still not posted.)
- [x] **P0-5 Champion bookkeeping + gate script.** ✅ DONE 2026-06-16. `results/champion.txt` (init: `v001_random`).
      `scripts/gate.sh <candidate_version> [n_games] [seed]`: Docker gate runner candidate-vs-champion.
      `scripts/gate.sh v002_lucario 300 42` → PASS (292/300 wins, 97.3%, Wilson CI [0.9483, 0.9864]).
      `scripts/promote.sh <version>`: updates `champion.txt` (refuses if submission dir missing).
      `src/eval/gate_runner.py`: loads both agents via importlib with CWD trick for deck.csv.
      **Done when:** `./scripts/gate.sh` prints a correct PASS/FAIL verdict. ✅

---

## Phase 1 — v002 = best sample deck + reference pilot (bank the cheap win FAST)

> Two free wins at once: our v001 used the WEAKEST deck (Mega Abomasnow, 40.2%) piloted by RANDOM.
> Switch to the BEST sample deck (**Mega Lucario ex, 60.4%**) with its **reference rule-based agent** — both already
> saved in `data/sample_agents/mega_lucario_ex/{main.py, deck.csv}`. This becomes the first real champion AND the
> rollout/eval policy reused by search/self-play later.

- [x] **P1-1** ✅ DONE 2026-06-16. Created `submissions/v002_lucario/` (main.py + deck.csv + cg/). Renamed original
      `agent` to `_agent_impl`; added never-throw `agent()` wrapper. Clean importable copy at
      `src/agents/heuristic_agent.py` (loads deck from co-located deck.csv via `__file__`, falls back to CWD/Kaggle path).
- [x] **P1-2** ✅ DONE 2026-06-16. Deck legality: `battle_start` returned valid obs_dict (errorPlayer=-1) — Lucario deck is legal.
      Smoke: 20 games, ZERO exceptions, v002 won 19/20 (95.0%). Full gate 300 games: 292W–8L–0D,
      winrate=0.9733, Wilson CI=[0.9483, 0.9864] — **PASS all gate conditions**.
- [ ] **P1-3** Submit via gate, `promote.sh` after ladder confirms. Update `results/submissions.csv` + CLAUDE.md roadmap.
      **Done when:** v002 is the champion and μ is meaningfully above 603.6.
- [ ] **P1-4 (optional, parallel):** also pull the other 3 sample agents/decks (Dragapult ex = 2nd best 55.6%, Iono,
      Abomasnow) into `data/sample_agents/` — they feed the Phase 3 deck gauntlet and serve as benchmark opponents.
      (Lucario notebook: `kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck`; find the sibling notebooks in forums.)

---

## Phase 2 — v003 engine search (determinized MCTS) — main strategic lift

> The engine's `search_begin/search_step/search_end` IS the forward simulator. Build search on top of it.
> Hidden info (opp deck/hand/prizes/active) is sampled (determinization); search the resulting perfect-info game.

- [x] **P2-1** ✅ DONE 2026-06-17. `src/agents/search_agent.py`: determinized rollout search. Samples K hidden
      states (our deck = known 60 minus visible, split into deck/prize; opponent prior = Lucario list), `search_begin`
      per determinization, rolls out each root option with the P1 heuristic policy to terminal, aggregates, picks best.
      Never-throw wrapper + heuristic fallback. `src/agents/heuristic_agent.py` gained `decide_on_obs()` (rollout policy).
- [ ] **P2-2** Tune K/depth/budget. **NOT DONE — this is the gap.** The gate showed only **8 ms/move actual** vs a
      ~2.5 s/decision budget → the search is far too shallow to beat the heuristic. Need: bigger K, deeper rollouts /
      proper UCT selection, more rollouts per option (current per-option estimate is high-variance), better leaf eval.
- [ ] **P2-3** Gate vs champion. ❌ **FAIL 2026-06-17.** `submissions/v003_search/` (re)packaged + gated fresh:
      vs v002_lucario, 300 games, seed 42: **157W–143L–0D, winrate 0.5233, Wilson CI [0.4669, 0.5792]** — fails the
      ≥55% bar and Wilson_lo>0.50; statistically indistinguishable from the heuristic it rolls out. 0 exceptions,
      8.2 ms/move, 359s wall (~1.2s/game). (A prior subagent run measured 48.67% on the same seed; both runs agree the
      edge is ~0 — engine coin flips aren't seeded by our `seed`, so there's run-to-run noise around 50%.) NOT submitted
      (gate FAIL → no submit). Need P2-2 first. **Done when:** v003 beats v002 ≥55%/300 and climbs the ladder.

---

## Phase 3 — Deck optimization via gauntlet (all 3 sources at once) — v004

> Unify the three deck sources into one **deck gauntlet**: round-robin self-play with our best pilot; best deck wins.

- [ ] **P3-1 Source A (tune current):** variants of the Mega Abomasnow/Kyogre list (energy counts, tech cards).
- [ ] **P3-2 Source B (net-deck):** from P0-4, reconstruct strong opponents' decks (fill gaps from card pool by archetype).
- [ ] **P3-3 Source C (from scratch):** hand-build 1–2 known-strong archetypes from `EN_Card_Data.csv` (respect legality).
- [ ] **P3-4 Gauntlet:** `src/eval/gauntlet.py` round-robins all candidate decks (each piloted by the current best agent),
      ranks by aggregate winrate. Pick the winner; sanity-check vs the meta seen in replays.
- [ ] **P3-5** Swap winning deck into the champion pilot, gate, submit `submissions/v004_deck/`, promote on ladder win.
      **Done when:** the gauntlet-winning deck + best pilot is the champion.

---

## Phase 4 — Imitation learning from replays — v005 (ML begins)

> Frame play as ranking the available `select.option`s. Train on (obs, chosen-option) pairs (P0-4),
> prioritizing **top-player** replays. Inference must fit Q-DEPS + Q-TIME (prefer LightGBM ranker or a small
> numpy MLP exported to dependency-light inference; use torch only if Q-DEPS allows it in the submission runtime).

- [x] **P4-1** ✅ DONE 2026-06-17. `src/il/features.py`: 49 features per option — 17-dim type one-hot, 16 local
      (attack damage/KO/prize/weakness, card type/hp/retreat, retreat HP ratio, is-end), 16 global (turn, prizes,
      hand, bench, active HP, energy count, flags).
- [x] **P4-2** ✅ DONE 2026-06-17. `src/il/dataset.py`: built from 204 games (189 top-players + 15 ours),
      19,440 (obs,action) pairs; skill-weighted by player μ; train/val split 85/15.
- [x] **P4-3** ✅ DONE 2026-06-17. `src/il/train.py`: LightGBM lambdarank, 121 trees. Offline val top-1
      overall=0.583, MAIN=0.534 vs opt-0 baseline 0.457/0.378. Top-3 overall≈0.79. Model at `models/il_ranker.txt`.
- [x] **P4-4** ✅ DONE 2026-06-17. Packaged `submissions/v004_il/` (main.py inlining IL scorer + heuristic fallback,
      features.py, il_ranker.txt, EN_Card_Data.csv, deck.csv, cg/). Model ran 100% of decisions (5022/5022, 0
      heuristic fallbacks, 0 exceptions). **Smoke gate 100g seed 42 vs v002_lucario: 31W–69L, winrate=0.310,
      Wilson CI [0.228, 0.406], 0.177 ms/move → GATE FAIL.** Per smoke-skip rule (< 0.48), 300g gate not run.
- [x] **P4-5** ✅ DONE 2026-06-17. Built `submissions/v005_hybrid/` — Lucario heuristic scores + IL ranker scores,
      rank-normalized then combined: `combined[i] = (1-λ)*H_norm[i] + λ*IL_norm[i]`. λ read from `HYBRID_LAMBDA` env var.
      λ sweep (100g, seed 42) vs v002_lucario:
        λ=0.00 (sanity) → 49W-51L, WR=0.490 ✓ (pure heuristic ≈ 50%, harness correct)
        λ=0.15 → 44W-55L-1D, WR=0.444, IL_override=0.5%
        λ=0.30 → 51W-49L, WR=0.510 (best), IL_override=3.9%
        λ=0.50 → 36W-64L, WR=0.360, IL_override=17.3%
      Best λ=0.30 at WR=0.510 < 0.52 threshold → 300g gate skipped. GATE FAIL. Champion stays v002_lucario.
      Conclusion: mixing mismatched IL into the heuristic HURTS monotonically with λ. See Status Log 2026-06-17 (h).
- [x] **P4-6** ✅ DONE 2026-06-17. Data-matched IL: retrained on Crustle-only expert games → `submissions/v006_crustle/`
      (Dwebble/Crustle deck + `il_crustle.txt`). 4983 MAIN decisions, 20 trees (early-stop). Offline val MAIN top-1=0.548
      vs option-0=0.578 (model below option-0 offline; confirmed fine in live play). Gate 300g seed 42 vs v002_lucario:
      **259W–41L–0D, WR=0.8633, Wilson CI [0.820, 0.898] → GATE PASS.** 0 exceptions, 100% model decisions.
      Awaiting human approval to submit. See Status Log 2026-06-17 (i).

---

## Phase 5 — Opponent modeling — sharpen search & IL

- [ ] **P5-1** Infer opponent deck/archetype from revealed cards during a match (from `logs` + visible board);
      maintain a belief over their remaining deck/hand for better determinization in P2 and better priors in P4.
- [ ] **P5-2** Feed beliefs into search_agent's hidden-state sampling. Gate; submit `submissions/v006_oppmodel/` if it wins.

---

## Phase 6 — Self-play RL / expert iteration — v007+ (highest ceiling)

> Use the engine search (P2) + a learned policy/value net (init from P4) in an **AlphaZero-lite / expert-iteration**
> loop: search produces improved action targets → train net → repeat. Run batches on Kaggle notebooks (9h sessions);
> Docker only for correctness. Pure PPO self-play only if infra throughput allows the volume.

- [ ] **P6-1** Value/policy net + training loop `src/rl/` (selfplay → targets → train → eval-gate vs champion).
- [ ] **P6-2** Iterate; only promote nets that pass the gate. Keep the search/heuristic fallback inside the agent.
      **Done when:** an RL-refined agent is the champion. (Stretch goal; respect the deadline freeze above all.)

---

## Decision rules for different outcomes (the "what if")

| Situation | Action |
|---|---|
| Candidate fails the gate (<55% or CI not clear) | Do NOT submit. Iterate or discard. Keep champion. |
| Candidate passes gate but ladder μ drops after ≥10 episodes | Roll back: re-submit champion. Diagnose via episode replays. |
| Engine per-move time limit (Q-TIME) is tight | Lower search K/depth; precompute; fall back to heuristic. Profile in P0-2. |
| `torch` not importable in submission runtime (Q-DEPS) | Use LightGBM / numpy-only inference; export model accordingly. |
| Docker on ARM too slow for big gauntlets | Run volume self-play on Kaggle notebook; Docker only for smoke tests. |
| Meta shifts (new strong opponents in replays) | Re-run P3 gauntlet with new net-decked candidates; adapt deck. |
| Out of submission budget for the day | Stop submitting; keep iterating locally; resume tomorrow. |
| Within 5 days of deadline (≤2026-08-11) | FREEZE experiments. Ensure last 2 = champion + best vetted variant only. |
| A phase stalls / is too costly for the gain | Skip forward; ML/RL is the ambition but never at the cost of a stable last-2. |

---

## Status Log (append-only; newest on top)

- **2026-06-17 (j)** — BUG FIX: v006 sub 53775584 FAILED Kaggle self-validation (ep 80310903) with
  `NameError: name '__file__' is not defined`. Root cause: Kaggle loads agents via `exec(code_object, env)`
  where `__file__` is NOT injected into the namespace; line 26 of main.py used it unconditionally at import
  time. Local gate couldn't catch it because `gate_runner.py` uses `importlib` which DOES set `__file__`.
  Fix applied to `submissions/v006_crustle/main.py` and `features.py` (and also retroactively to v004/v005):
  `_DIR` now resolves in order (1) `/kaggle_simulations/agent` if that dir exists, (2) `__file__`-based
  parent (try/except NameError), (3) `os.getcwd()`. `features.py:_find_card_data_csv()` received same guard.
  Verification: `exec(compile(src, "main.py", "exec"), {"__name__": "__main__"})` (no `__file__` in ns)
  from submission CWD → no NameError, `_DIR` = submission dir, `agent({"select": None})` returned 60 ints →
  **PASS**. Regression gate 100g seed 42 vs v002_lucario: **86W–14L, WR=0.860, model=100%, 0 fallback,
  0 exceptions** (identical to pre-fix). DO NOT RESUBMIT without explicit human approval.

- **2026-06-17 (i)** — Phase 4 data-matched IL (Crustle): built `submissions/v006_crustle/` (Dwebble/Crustle deck,
  LightGBM lambdarank trained on Crustle-only expert games).
  **Recon finding:** Dwebble/Crustle is 55% of top-player deck-slots (222/378) with 80%+ win rate; our
  Lucario deck is only ~10%-winrate among top players → deck switch + data-matched IL is the right lever.
  **Training:** 222 Crustle slots from 189 replays → 6957 decisions (4983 MAIN), 85/15 split by episode.
  Offline val top-1: model overall=0.520 / MAIN=0.548 vs option-0 baseline 0.656 / 0.578. Model below
  option-0 offline (option-0 biased in Crustle Pokémon-heavy contexts), but what matters is live play.
  Model ran 100% of decisions (5677/5677, 0 exceptions).
  **Smoke gate (100g, seed 42) vs v002_lucario: 86W–14L, WR=0.860, Wilson CI [0.779, 0.915] → proceed.**
  **Full gate (300g, seed 42) vs v002_lucario: 259W–41L–0D, WR=0.8633, Wilson CI [0.820, 0.898] → GATE PASS.**
  PASS criteria: WR≥0.55 ✓, n≥300 ✓, wilson_lo>0.50 ✓ (all three). Mean move time 0.138 ms.
  Deck/model at `submissions/v006_crustle/`. AWAITING HUMAN APPROVAL TO SUBMIT.

- **2026-06-17 (h)** — Phase 4 P4-5 DONE: built `submissions/v005_hybrid/` (heuristic-IL blend). Design:
  heuristic scores + IL (LightGBM) scores each rank-normalized → combined[i] = (1-λ)*H_norm + λ*IL_norm;
  selection = top-k by combined; λ from `HYBRID_LAMBDA` env var (default 0.2); λ=0 → pure heuristic.
  gate.sh patched to forward `-e HYBRID_LAMBDA` into Docker. λ sweep, 100g seed 42 vs v002_lucario:
    λ=0.00 (sanity) → 49W-51L, WR=0.490 ✓ (pure heuristic mirror ≈ 50% — harness correct, 0 exceptions)
    λ=0.15 → 44W-55L-1D, WR=0.444, IL_override=0.5%, 0 exceptions
    λ=0.30 → 51W-49L, WR=0.510 (best), IL_override=3.9%, 0 exceptions
    λ=0.50 → 36W-64L, WR=0.360, IL_override=17.3%, 0 exceptions
  Best λ=0.30 WR=0.510 < 0.52 smoke threshold → 300g gate skipped. GATE FAIL. Champion stays v002_lucario.
  One-line verdict: mixing the mismatched IL (trained on Abomasnow/Kyogre games) into the Lucario heuristic
  HURTS — monotonically worse with more IL weight. The IL's 0.310 WR in v004 drags every hybrid down;
  even 3.9% override rate at λ=0.30 is enough to underperform the pure heuristic. Remedy options: (1) retrain
  IL on Lucario-only games, (2) deck-identity features in IL, (3) improve the heuristic pilot directly.

- **2026-06-17 (g)** — Phase 4 P4-4 DONE: built `submissions/v004_il/` (LightGBM IL ranker, never-throw with
  heuristic fallback). Docker image updated with lightgbm 4.6. Sanity: model loads in Docker (121 trees, 49 features),
  deck-select correct. Model ran **100% of decisions** (5022/5022 model, 0 heuristic fallback, 0 exceptions) → no
  silent fallback. **Smoke gate (100g, seed 42) vs v002_lucario: 31W–69L, winrate=0.310, Wilson CI [0.228, 0.406],
  mean move time 0.177 ms → GATE FAIL.** Smoke is well below 0.48 threshold — 300g gate skipped per policy.
  Champion stays v002_lucario. Root-cause hypothesis: the IL ranker trained on a mix of top-player games across all
  deck archetypes (Lucario=minority there — most top players run Abomasnow/Kyogre), but v004_il plays the Lucario
  deck. The model likely learned Abomasnow-specific patterns and gives wrong advice for Lucario-specific decisions
  (energy attachment, boss-orders target, retreat timing). The heuristic already has all this Lucario-specific
  knowledge hand-coded. **Next steps:** (1) filter training data to Lucario games only, OR (2) add deck-identity
  features so the model distinguishes archetypes, OR (3) pursue P4-5 hybrid (IL for rough ranking, heuristic
  override for the few high-stakes MAIN decisions). Alternatively, pivot to improving the heuristic pilot directly.

- **2026-06-17 (f)** — Recon + Phase-4 START. Human asked to review standing first: μ=798.8, **RANK 138/570** (top ~24%),
  v002 ladder record ~7W-5L over 12 games, μ STALLED since 03:58. **LB inflated hard** in 1 day (top20 810→1005), so
  standing still = sliding. Losses spread across many decks → no hard counter; bottleneck is PILOT quality, not deck.
  Forum: inference-env thread 708810 still unanswered; organizer top-episode export not posted yet. Human chose to
  CLOSE Q-DEPS first → **Q-DEPS ANSWERED (P0-3, free sources, no probe)**: torch(CPU)/lightgbm/sklearn/xgboost all in
  the runtime image → IL unblocked. Human then chose **Phase 4, data-first**. **P0-4 DONE**: harvested 189 top-player
  replays (μ 978–1160) + 15 ours = 204 games; `src/replays/parse.py` (macOS, no libcg) → **19,440 (obs,action) pairs
  (7,721 winner)**, MAIN 62%. NEXT: build the IL pipeline (P4-1 features + P4-2 dataset + P4-3 LightGBM ranker) and
  measure OFFLINE top-1 accuracy on held-out expert moves as a cheap checkpoint BEFORE investing in agent+gate.
  (All execution delegated to Sonnet subagents per the new rule.) Champion stays v002_lucario.

- **2026-06-17 (e)** — Phase 3 START — deck gauntlet built + SMOKE run → deck lever looks TAPPED with the current pilot.
  Human chose Phase 3 over IL/search. KEY CONSTRAINT confirmed: the pilot is HARD-CODED to Mega Lucario ex, so the
  gauntlet can't "pilot any deck" — it tunes the Lucario list with the same pilot. Built `decks/gauntlet/_gen_decks.py`
  (pilot-SAFE recount-only variants), `src/eval/gauntlet.py` (round-robin matrix + winrate-vs-base + Wilson, fresh pilot
  module per side to avoid the plan/pre_turn/ability_used global collision), `scripts/gauntlet.sh`. NEW FACT: Hero's Cape
  (id 1159) is **Ace Spec → max 1 copy** (engine errorType=4); recorded in Guardrails. SMOKE (5 decks, 10 pairs, 40
  games each, seed 42): 0 exceptions, 0.15 ms/move. NO variant beats 00_base with Wilson_lo>0.50 — best was 03_cape2
  (→Switch×3/−1 energy) at 0.575 vs base but Wilson_lo=0.422 (n=40 noise ±15%). All variants' CIs straddle 0.50 →
  statistically indistinguishable from base. CONCLUSION: recount-only deck tuning under a frozen Lucario pilot won't
  clear the gate — did NOT burn 300-game gates. The real ceiling now needs a BETTER PILOT (champion is already near
  top-20 at μ=798.8 with the sample pilot) → Phase 4 (IL from top-player replays) or improving the heuristic pilot.
  Decision pending human. Champion stays v002_lucario.

- **2026-06-17 (d)** — P2-2 search SCALING tried → VALIDATED ≈ no gain → PIVOT recommended. A Sonnet subagent scaled
  the search (env-configurable budget V003_*, UCB1 root selection, budget-driven rollouts with fresh determinizations,
  bank-safe self-limiting) but died mid-run (auth/403) before validating + bookkeeping. The scaled search genuinely
  runs now (a 20g smoke took ~32 min, ~96 s/game — vs the naive 8 ms/move that barely searched, confirming the naive
  version mostly silently fell back to the heuristic). Independent re-check by main: rebuilt image + gate v003 vs v002,
  30 games seed 1, cap 2.0s/decision → **14W-16L = 46.7%, mean 79 ms/move → FAIL.** Combined with the two naive 300g
  gates (48.7%, 52.3%), ALL THREE sit at ~50%: a 1-ply determinized rollout search whose rollouts ARE the heuristic
  ≈ the heuristic (few true MAIN forks/game; the heuristic already plays them well; 1-ply = one weak policy-improvement
  step). CONCLUSION: this search doesn't pay off. v003 NOT submitted. Champion stays v002_lucario (μ live on ladder).
  **RECOMMEND PIVOT → Phase 3 (deck gauntlet) or Phase 4 (IL).** Caveat: vendored main.py (02:30) is ~11 min older than
  dev search_agent.py (02:41); the gate tested the vendored copy. Artifacts kept (search_agent.py, search_diag*.py).

- **2026-06-17 (c)** — Phase 2 RESUME. The earlier `submissions/v003_search/` was gone, so re-packaged it from scratch
  (self-contained `main.py` inlining heuristic + determinized rollout search, never-throw wrapper, Lucario `deck.csv`,
  `cg/`). Re-ran Q-SEARCH probe in Docker (`./scripts/probe.sh 20 7`): search_begin 0.30 ms, search_step 0.22 ms/step,
  ~88 steps/rollout (max 108), ~52 rollouts/sec, ~129 rollouts/decision feasible (K≈10 × 12 opts) — matches (b).
  **Full gate 300g vs v002 (seed 42): 157W–143L, winrate 0.5233, Wilson [0.4669, 0.5792] → FAIL** (need ≥55% &
  Wilson_lo>0.50). 0 exceptions, 8.2 ms/move, 359s wall. Confirms (b): naive 1-ply + heuristic-rollout search is ~50/50
  vs the heuristic it rolls out — the bank is 99.7% unused (8 ms of ~2500 ms/decision). v003 NOT submitted (gate FAIL).
  Champion stays v002_lucario. RECOMMENDATION: don't burn a submission on v003-as-is. Either invest in P2-2 (much
  bigger K + multiple rollouts/option + UCT + a real leaf eval; the bank allows 30–50× more compute) OR pivot to
  Phase 3 (deck gauntlet) / Phase 4 (IL), which likely have a better effort:gain ratio. Decision pending human.

- **2026-06-17 (b)** — Phase 2 attempt (Sonnet subagent). Q-SEARCH ANSWERED via `src/eval/search_probe.py`
  (+`scripts/probe.sh`): ~52 rollouts/sec emulated, ~129 rollouts/decision feasible. Built `src/agents/search_agent.py`
  (determinized rollout search; opp prior = Lucario) + `decide_on_obs()` in heuristic_agent. Packaged
  `submissions/v003_search/` (self-contained main.py, never-throw, Lucario deck, cg/). Smoke 40g = 65% (noisy small n).
  **Full gate 300g vs v002: 146W–154L, winrate 0.4867, Wilson [0.4306,0.5430] → FAIL.** 0 exceptions, only 8 ms/move
  (huge unused bank). Conclusion: naive shallow rollout search ≈ the heuristic it rolls out — needs P2-2 (much bigger
  K/depth, UCT, more sims/option, better leaf eval). v003 NOT submitted (gate FAIL). Champion stays v002_lucario.
  NEXT: tune the search (P2-2) OR pivot. v002 keeps playing the ladder (μ live).

- **2026-06-17 (a)** — v002_lucario won 2 public games (vs Kaiman2110 ep80213820; vs "Pikachu I choose you!" ep80214311,
  133 turns) → **μ 600→682.1→779.9**, promoted to champion (results/champion.txt). v001 retired (μ=483.8, 1W-3L).

- **2026-06-16 (e)** — P0-5 + P1-1 + P1-2 DONE (subagent run). Built: `results/champion.txt` (v001_random),
  `src/eval/gate_runner.py` (importlib CWD trick, PASS/FAIL verdict), `scripts/gate.sh`, `scripts/promote.sh`,
  `submissions/v002_lucario/` (main.py with never-throw wrapper + deck.csv + cg/), `src/agents/heuristic_agent.py`.
  Deck legality: `battle_start(v002_deck, v001_deck)` → valid obs (errorPlayer=-1). Smoke 20 games: 19W–1L–0D, ZERO exceptions.
  Full gate 300 games (seed=42): **292W–8L–0D, winrate=0.9733, Wilson CI=[0.9483, 0.9864] → GATE PASS**.
  v002 NOT submitted (awaiting human approval). NEXT: P1-3 — submit v002_lucario → promote on ladder win.

- **2026-06-16 (d)** — P0-1 DONE (pilot via Sonnet subagent). Docker harness works: `src/sim_harness.py`
  (`tournament` + Wilson CI + timing), `docker/Dockerfile`, `.dockerignore`, `scripts/sim.sh`. 20-game smoke:
  0.600 winrate, 0.089 ms/move. Q-LEGAL verified (ValueError on bad length; null battlePtr + errorType=3 on illegal
  composition). Pilot confirmed the planner is followable by a cheap model. NEXT: P0-5 gate scripts → P1 v002_lucario.
- **2026-06-16 (c)** — Pulled top-1 player's replay (net-deck access via `team-submissions`→`episodes`→`replay` works;
  saved to `data/episodes/top_players/`). KEY: #1 `platypus2000jp` (μ=1030) plays a **tuned Abomasnow** (our archetype)
  → piloting/tuning > deck choice; don't discard Abomasnow. Recorded hackathon-input note. Files/memory audited complete.
- **2026-06-16 (b)** — Forum/env deep-dive. Answered Q-TIME (600s bank/game), Q-LEGAL (60 cards + engine validates),
  Q-RULES (minor diffs, own decks allowed). Learned: 4 sample decks, **our v001 deck is the WEAKEST (40.2%)**,
  **Mega Lucario ex is best (60.4%)** → saved its reference agent+deck to `data/sample_agents/mega_lucario_ex/`.
  Crash/timeout/invalid = instant loss. New episodes parsed: v001 went **1W–2L** public (W vs porply; L vs
  Kazuyoshi SUZUKI, taichi-1). Reworked Phase 1 → v002 = Lucario deck + reference pilot. Still open: Q-DEPS, Q-SEARCH.
  NEXT: P0-1 (Docker harness), then P1 (ship v002_lucario).
- **2026-06-16 (a)** — Plan created. v001_random is champion (μ=603.6). Deck = Mega Abomasnow ex / Kyogre water control.
