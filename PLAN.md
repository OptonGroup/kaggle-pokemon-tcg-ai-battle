# PLAN — Pokemon TCG AI Battle (execution planner)

> **This file is the single source of truth for execution.** A junior model reads it,
> finds the first unchecked task, does it, then updates the **Status Log** at the bottom.
> Read `CLAUDE.md` first for project facts; this file is the *what to do next*.

Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
Sim deadline: **2026-08-16** → ~2 weeks extra games → final ~2026-09-01.
Current standing (2026-06-16): v001_random μ=**603.6**; leaderboard top-1 ≈ 1030, top-20 ≈ 810.

---

## 0. How a session works (read every time)

1. Open this file. Find the **NEXT ACTION** pointer (just below) and the first unchecked `[ ]` task.
2. Do exactly that task. If it's heavy/mechanical, spawn a **lighter subagent** for it.
3. Before any submission, run the **Submission Gate** (§Guardrails). No gate pass → no submit.
4. Append a dated entry to the **Status Log** (bottom). Update checkboxes. Update `results/submissions.csv`.
5. If you discover a fact worth keeping across sessions, write it to memory (see CLAUDE.md memory rules).

**NEXT ACTION:** P1-3: submit v002_lucario (awaiting human approval) → promote on ladder win.
P0-4 (replay tooling) and P0-2/P0-3 (Kaggle timing + deps probe) can run in parallel while waiting.

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
- **Q-DEPS:** Which libs import in the Kaggle submission runtime (`torch`/`lightgbm`/`onnxruntime`/`sklearn`)? → P0-3.
- **Q-SEARCH:** Real cost of one `search_begin`→`search_step` cycle; depth/`K` feasible within the 600s bank. → P2.
  (Data point from P0-1: a plain engine `select` is ~0.089 ms even under ARM emulation, so the bank is huge relative
  to engine cost — search budget will be dominated by our own sampling/eval, not the engine.)

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
- [ ] **P0-3 Submission-runtime deps probe.** In a throwaway submission (or the validation log), print
      `import`ability of `numpy, pandas, lightgbm, torch, onnxruntime, sklearn`. Record Q-DEPS.
      **Done when:** we know the allowed inference stack for ML phases.
- [ ] **P0-4 Replay tooling.** `src/replays/parse.py`: from `data/episodes/*-replay.json` extract
      (a) `(observation, chosen_option_indices)` training pairs from every step where `select != null`,
      (b) each player's revealed deck/cards. Helper `scripts/fetch_episodes.sh` already exists — extend to also
      pull organizers' published top-rated episode exports when available.
      **Done when:** parsing existing episodes yields a tidy list of (obs, action) pairs + a per-player card multiset.
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

- [ ] **P2-1** `src/agents/search_agent.py`. For the current decision: sample K plausible hidden states
      (use P0-4 card knowledge + simple priors), run the engine search per determinization with a **time budget**
      below Q-TIME, score outcomes with the P1 heuristic as the rollout/eval, aggregate, pick the best option set.
      Start simple (1–2 ply greedy + heuristic eval), then iterative-deepening MCTS if budget allows.
- [ ] **P2-2** Tune K (determinizations), depth, and budget so the worst-case move stays under Q-TIME with margin.
      Add a hard fallback: if over budget, return the P1 heuristic move.
- [ ] **P2-3** Gate vs champion. Package `submissions/v003_search/`, submit through the gate, promote on ladder win.
      **Done when:** v003 beats v002 ≥55%/300 and climbs the ladder.

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

- [ ] **P4-1** Featurization `src/il/features.py`: per-option features (type, target Pokémon hp/energy, attack id/damage,
      is-KO, is-supporter, retreat cost…) + global state features (turn, prizes left both sides, board, hand size).
- [ ] **P4-2** Dataset build `src/il/dataset.py` from all available replays; weight by player skill/score.
- [ ] **P4-3** Train `src/il/train.py` a per-option ranker; for multi-select, pick the top-k legal set.
      Export to the lightest inference format allowed by Q-DEPS.
- [ ] **P4-4** `src/agents/il_agent.py` loads the model; offline accuracy vs held-out expert moves, then gate vs champion.
- [ ] **P4-5** Optionally **hybrid**: IL policy to order/prune moves, engine search to verify → faster + stronger.
      Package `submissions/v005_il/`, gate, submit, promote on ladder win.
      **Done when:** IL (or IL+search hybrid) beats v004 ≥55%/300.

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
