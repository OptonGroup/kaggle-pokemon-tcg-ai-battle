# PLAN — Pokemon TCG AI Battle (execution planner)

> **This file is the single source of truth for execution.** A junior model reads it,
> finds the first unchecked task, does it, then updates the **Status Log** at the bottom.
> Read `CLAUDE.md` first for project facts; this file is the *what to do next*.

Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
Sim deadline: **2026-08-16** → ~2 weeks extra games → final ~2026-09-01.

## Current standing (2026-07-11) — plan v4 (HYBRID PIMC, user-confirmed)

- **VECTOR A confirmed by user 2026-07-11 (AskUserQuestion):** аудит реплеев (off-by-one) + retrain prior →
  анкерный оценочный пул → PIMC v2 с раздельными rollout-политиками → live-probe → опционально ExIt-lite /
  deck-swap. Фазы = **§H** ниже. §S закрыт (07-05), §F2–F4 superseded.
- **Live (2026-07-11):** tracked-2 = {v018_grimmsnarl **484.1** (1W-6L последние 7 игр), v007_crustle **733.1**};
  team **#1790/4773** (top 37.5%). Top-1 = Yushin Ito **1273.9–1346.0** (волатилен внутри дня), top-10 = **1101.6**,
  top-100 = **989.0**. Поле +16% за 8 дней. v009a заморожен на 724.1 (вытеснен из tracked-2 сабмитом v018 07-04).
- **Рейтинг живой и относительный:** байт-в-байт неизменный v007 сдрейфовал 930.7→733.1 (−197.6 за ~3 недели) —
  статичный агент тонет вместе с метой. Ре-сабмит НЕ наследует рейтинг (форум 714189).
- **Потолок BC ≈800–930 подтверждён трижды и независимо:** наш пик 930.7; kazuki-клон 453 при эксперте 1083
  (~40% силы); PokéForge-постмортем (форум 724187) — клон зрелой публичной политики (~802) = их ЛУЧШИЙ результат,
  все собственные архитектуры (turn-planning / self-play value-net / DAgger / A2C) провалились на ладдере.
  Топ-100 (989) BC-методами недостижим.
- **Топ-1 по таймингам = тяжёлая модель + полный бюджет хода** (форум 724362, анализ 30k игр) → поиск на этом
  движке РАБОТАЕТ в бою; ниша «думающих» агентов почти пуста (основное поле отвечает ~0.03 c/ход).
- **⚠ Форум 717141 (07-09, организатор молчит): возможный off-by-one в `selected` реплей-экспорта движка**
  (`entry[i].selected` = решение шага i−1; `entry[0].selected` всегда null) — ВСЕ наши (state, action)-пары IL
  под подозрением; это объясняло бы и «клон = 40% силы эксперта», и регрессию v008 от добора данных. Аудит = §H1,
  стоит часы, блокирует retrain.
- **Почему §S провалился и почему v4 — не повтор:** S4a mirror ≈0.50, S4b field −0.40/−0.15. Root cause —
  rollout ОБЕИХ сторон играла Crustle-IL (оппонент моделировался «Crustle-стилем»). §H3 чинит именно это
  (раздельные политики: свой prior + скриптовые пилоты архетипов оппонента) и добавляет два недостающих
  фундамента: чистые данные (§H1) и калиброванную локальную метрику (§H2), которых у §S не было.
- **Микро-факты:** первый ход +10пп (55.2/44.8, форум 723591; право выбора → первый в 91.5% случаев);
  48 игр/день/саб + 10% random matchmaking; число ядер Kaggle-воркера неизвестно — замерить в validation-эпизоде.

---

## Archive — Current standing (2026-07-05) — plan v3 (SEARCH PIVOT, superseded by v4)

- **STRATEGIC PIVOT 2026-07-05 (user-confirmed): STOP BC-cloning → build PIMC/search on Crustle.** See §S below.
- **Live scores now:** v007_crustle **760.4**, v009a_crustle **724.1** (both re-subs, decayed from 930/905 by meta
  rotation), v009b (old) still **910.2**. **v018_grimmsnarl = 453.5 LIVE (💀 below random 481)** — the decisive data
  point: kazuki0123 (the cloned expert) LIVES at 1083, our clone of his exact deck = 453 → **BC captures only ~40% of
  expert skill even on a "BC-friendly midrange" deck**. Top-1 = Majkel1337 **1223**; top-20 ≈ 1065; kazuki0123 #10=1083,
  aidy #16=1073. tracked-2 currently = {v018 453, v007 760} — user chose NOT to evict v018 (July rank cosmetic;
  freeze is 08-11; restore champions before then).
- **THE CEILING IS THE METHOD, NOT US.** Public SOTA (repo `wmh/ptcg-abc`) best rule-based = **836** (Bellibolt);
  our Crustle IL peaked **930** — we already beat public heuristic/IL knowledge. But BC/IL caps ~800–930 for everyone
  (imitation can't exceed the expert, and drifts off-corridor). BC failed on Iono(336-507)/Alakazam(gate 43%)/
  Cubchoo(BC-dead)/Grimmsnarl(453). **930→1223 is unreachable by more BC.** The untapped lever = the engine's own
  `search_begin/search_step` API + native engine (4200 rollouts/turn, we use 8ms of a 2500ms/decision budget). Past
  search failed only because it was 1-ply with a weak rollout policy (≈ itself). Now we have a strong prior (Crustle
  IL) + fast native engine → PIMC is the credible path above 930. [[project-bc-combo-failure]]
- **META ROTATED (recon 2026-07-03):** top-10 has NO Crustle/Iono/Lucario/Abomasnow. Dominant = Marnie's
  Grimmsnarl ex / Munkidori Darkness toolbox (ranks 1,3,10), plus Cubchoo/Water toolbox package (3/10),
  Mega Starmie ex, Alakazam, Comfey/Chandelure. Staples: Fezandipiti ex, Dunsparce/Dudunsparce. Ex-#1 onechan1
  (Iono) → rank 206, inactive since 06-17. Meta rotates ~every 2 weeks → re-recon before archetype commitments.
- **Platform news (organizer, 06-30/07-01):** matchmaking = 48 games/day/submission + 10% random-opponent matching;
  released native binaries (libcg.dylib macOS, libcg-arm64.so) AND full engine source (ptcg_engine.zip on Data tab;
  license = local testing/verification only, NO exploiting engine bugs). Docker-emulation-only constraint is gone
  (Docker still required for final pre-submit validation: submission runtime = linux x86_64).
- **Key implication:** final leaderboard = strength of the tracked latest-2 as of the freeze; with 48 games/day even
  a late submission converges (~600+ games in the 2-week post-deadline window). July public rank is cosmetic;
  live probes are cheap (~100-150 games in 2-3 days).

---

## 0. How a session works (read every time)

1. Open this file. Find the **NEXT ACTION** pointer (just below) and the first unchecked `[ ]` task.
2. Do exactly that task. If it's heavy/mechanical, spawn a **lighter subagent** for it.
3. Before any submission, run the **Submission Gate** (§Guardrails). No gate pass → no submit.
4. Append a dated entry to the **Status Log** (bottom). Update checkboxes. Update `results/submissions.csv`.
5. If you discover a fact worth keeping across sessions, write it to memory (see CLAUDE.md memory rules).

**NEXT ACTION:** **PIMC ЗАКРЫТ ОКОНЧАТЕЛЬНО (§H3-4-re KILL, честный, 700 игр). Новый вектор: (1) дек-viability — наши деки vs расширенное мета-поле (16 top-cut дек из recon), решить какая дека жива в текущей мете; (2) на выжившую деку — улучшение политики, начиная с дешёвого ExIt-lite (каркас `src/exit/exit_loop.py` готов, F3-3 smoke не прогонялся) перед дорогим PPO (+65 ELO, recon). Развилка по вектору — у пользователя.**
§H1 закрыт (NOT_SHIFTED — IL-данные чисты, retrain не нужен). §H2 закрыт: пул+калибровка построены
(6300 игр, ~50 игр/с), но LOO Spearman 0.714 < 0.8 ⇒ predicted-μ = ранжировщик кандидатов, НЕ прогноз live;
финальный арбитр — live-probe. §H0-1 (git-коммит копилки) всё ещё ждёт команды человека.
**CRITICAL (действует и в v4): search-сабмиты ОБЯЗАНЫ бандлить НОВЫЙ пересобранный `libcg.so` (SHA `ffd89bf9…`)
из `data/engine/sample_submission_cg/cg/`** — старый `.so` рассинхронизирован с платформой → результаты поиска
нерелевантны. [[project-native-engine]]

---

## Guardrails (NON-NEGOTIABLE)

- **Submission Gate.** Only submit a candidate if ALL hold:
  1. **Same-deck iterations** (candidate plays the champion's archetype): it beats the current **champion** in local
     self-play: winrate **≥ 55%** over **≥ 300 games** (≥150 going first, ≥150 second), AND Wilson 95% lower bound
     **> 50%**. (Run big gauntlets on Kaggle notebook for speed; Docker for quick smoke checks.)
  2. **Candidates → Gauntlet-v3 = анкерный пул + predicted-μ (standard since 2026-07-11, см. §H2; допуск =
     predicted-μ кандидата > live-μ чемпиона + 50). Legacy Gauntlet-v2 (2026-07-03, F2-3):** opponent pool =
     {v007_crustle, v009a_crustle, lucario heuristic (v002), reconstructed top-meta decks, random}. PASS =
     ≥55% weighted overall AND no key matchup <45% (anti-v012 rule: no narrow counters) AND presubmit
     exec-no-`__file__` AND timing safe.
  3. It passes Kaggle **self-validation** (the validation episode on upload completes without error).
  4. We are **not** in the deadline freeze (see below).
  **Live probe (2-3 days, ~100-150 games) is the final arbiter — local gates systematically overestimated
  narrow counters (v012: 0.70 local vs 499 live).**
- **Champion bookkeeping.** `results/champion.txt` holds the current best version dir name.
  We must ALWAYS be able to re-submit the champion unchanged. Never delete a champion's dir.
- **Daily budget.** ≤ **5 submissions/day**. Reserve ≥1 slot for a champion rollback. Never burn all 5 on experiments.
- **Final-2 rule.** Only the **latest 2** submissions count for the final. So the last two uploads must be
  our **champion** + at most one well-tested variant. Never let an experimental/unvetted agent be one of the last two.
- **July tracked-2 policy (2026-07).** Experimental live probes are allowed — the July public rank is cosmetic.
  After each probe cycle, restore the champions to the tracked-2. What matters = strength of the pair at the
  freeze (08-11).
- **Matchmaking (since 06-30).** 48 games/day per submission + 10% random-opponent matching.
- **Deadline freeze.** From **2026-08-11** (5 days before sim deadline): no experimental submissions.
  Ensure the last two submissions are the champion (and the single best vetted variant). Stability > marginal gains.
- **Ladder rollback.** After a submit, watch episodes. If ladder μ trends **below** the champion after ≥10 public
  episodes, re-submit the champion (roll back) and investigate.
- **Engine is read-only.** Never edit files under any `cg/` dir. Copy `src/cg/` into each new `submissions/vNNN/` dir.
- **Native engine & source (since 06-30/07-01).** Organizer released native binaries (libcg.dylib macOS,
  libcg-arm64.so) + full engine source (ptcg_engine.zip on the Data tab). License = **local testing/verification
  ONLY — no exploiting engine bugs**. Docker (linux/amd64) remains **mandatory for final pre-submit validation**:
  the submission runtime is linux x86_64.
- **Format.** `.tar.gz` (NOT zip) via `tar -czvf submission.tar.gz *` from inside the submission dir.
  `main.py` at top level, plus `deck.csv` and `cg/`. Use `./scripts/submit.sh <version> [msg]`.
- **Deck legality.** Max 4 copies of any card by name (basic energy unlimited), exactly 60 cards, ≥1 Basic Pokémon.
  **Ace Spec = max 1 copy per deck** (engine rejects 2+ with `errorType=4`; confirmed 2026-06-17 with Hero's Cape id=1159).
  Validate every candidate deck before use.
- **Time bank.** Each agent has a ~600s bank for the WHOLE game (not per move). Budget search so the bank lasts
  every decision in a long game; keep a safety margin. Going over the bank = TIMEOUT = instant loss.
- **Never throw.** Any unhandled exception or illegal selection = instant loss. Wrap the agent in a try/except that
  falls back to a always-legal move (e.g. first `maxCount` indices) so a bug degrades gracefully instead of losing.
- **Human approval for every upload.** Never run `scripts/submit.sh` / `kaggle competitions submit` without the
  human's explicit submit approval ("залить"/"submit"/"да, грузи"); "continue" is NOT approval (CLAUDE.md rule 2).

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

> ⚠ **SUPERSEDED by 2026-07-03 meta rotation — see header; kept for reference.**

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

## Archive — Phases P0–P6 (2026-06-16..07-01), superseded by plan v2

> Compact summary; full details in the Status Log below + memory/. Old P0–P6 bodies removed 2026-07-03.

- **P0 infra DONE:** Docker self-play harness (`sim.sh`), gate runner (`gate.sh`/`promote.sh`), replay harvest+parse
  tooling (`src/replays/parse.py`, `src/eval/harvest_all_teams.py`), presubmit exec-no-`__file__` check, champion
  bookkeeping. The dev loop works.
- **P1 v002_lucario SHIPPED:** best sample deck + reference heuristic pilot → live 756→779.9.
- **P2 search v003 FAILED gate (~50%):** a 1-ply determinized rollout search ≈ its own rollout policy. Shelved.
- **P3 deck gauntlet:** recount-only tuning under a frozen deck-specific pilot = statistically indistinguishable
  from base. Shelved.
- **P4 IL — the PROVEN pipeline:** mixed-data v004/v005 failed; deck-matched Crustle IL v006/v007/v009a PASSED gates
  and scored **904.8 / 930.7 / 905.9 live**. v007 = best-ever (930.7).
- **Iono line v010a/v010c/v011/v012/v013/v014/v015 — ALL failed or regressed** (live 317–507): counter-deck bet
  invalidated by meta rotation + local counter-gates overestimating (v012: 0.70 local vs 499.2 live).
  Details in Status Log + memory (project_iono_pivot).
- **ExIt pipeline built 07-01** (`src/exit/`: pilot.py, pimc_agent.py, harness.py, exit_loop.py, opponents.py)
  **but NEVER executed** (audit 07-03). PIMC prototypes (`src/pilot/`) hard-coded to Iono → re-parameterize (F3-1).

---

## F0 — Stabilization (2026-07-03..04)

- [x] **F0-1 Re-submit champions unchanged.** ✅ DONE 2026-07-03 ~10:47 UTC: v009a first → submission ID **54295400**,
      then v007 → ID **54295405** (v007 = strongest stays newest in tracked-2). Both status PENDING self-validation
      as of writing — watch validation + first episodes, record scores in `results/submissions.csv` (rows added).
      Artifacts used (both presubmit PASS 2026-07-03): `results/v007_crustle.tar.gz` (byte-identical original),
      scratchpad `v009a_crustle_resub.tar.gz` (zero-diff fresh build; NOTE scratchpad is ephemeral — if lost, rebuild
      via `cd submissions/v009a_crustle && tar -czvf ... *` excluding `__pycache__`/`.DS_Store`).
      Purpose: restore tracked-2 + live signal "is Crustle alive in the new meta" within 2-3 days.
- [x] **F0-2a Bookkeeping** (this task). ✅ DONE 2026-07-03: PLAN.md v2, `champion.txt` → v007_crustle,
      `results/submissions.csv` updates (v010a/v015/re-submit rows, v012 live score), stray root episode files →
      `data/episodes/`, LB snapshot zip → `data/`.
- [ ] **F0-2b Native engine setup:** download ptcg_engine.zip + native libs, native macOS self-play, benchmark vs
      Docker (in progress by another agent 2026-07-03).
- [ ] **F0-3 Read forum threads:** "Leaderboard Scoring Inconsistency" (06-23, 62 votes), "Is Battle Simulate
      Matching Process Working Well?" (06-23, 56 votes) — extract any rating-mechanics facts.

## F1 — Meta recon → deck decision (by ~07-08)

- [x] **F1-1 Harvest fresh (post-06-30) replays** of top-100 teams (harvester exists: `src/eval/harvest_all_teams.py`;
      48 games/day → thousands of episodes). DONE 2026-07-03: top-100 harvested into `data/episodes/all_teams/`,
      per-team cap 8, meta in scratchpad `harvest_meta.json`.
- [x] **F1-2 Archetype clustering** (à la `src/eval/crustle_list_recon.py`): shares in top-10/50/100, matchup matrix
      from real game outcomes, stable-expert list per archetype. DONE 2026-07-03: Alakazam = top archetype
      (28% top-25, 34% top-50). CAVEAT (hit): `src/replays/parse.py` `revealed_cards()` mixes BOTH players' cards —
      use own-side filtering (own active/bench/hand/discard only, max-per-step then MAX across eps).
- [x] **F1-3 DECK DECISION by data:** DECK = **Alakazam**, primary expert = **aidy #8** (μ1127.4, wr0.89, ~1000 eps
      on Kaggle, deck stability 0.89, 8/8 eps reproduce core). Single-expert-first (aidy-only); fallback variants
      only if too narrow: Kohenyan#16 (Alakazam VAR — Dunsparce/Dudunsparce by diff IDs, +Neutralization Zone,
      +Wondrous Patch, no Night Stretcher) and Ajishio#13 (Alakazam VAR + Togepi→Togekiss 1/1 package). aidy IL
      volume: 1412 winner-MAIN pairs from 35 local eps (ceiling 1000 on Kaggle; ~15 more eps → ≥2000).

### Canonical aidy Alakazam 60-card decklist (reconstructed 2026-07-03, stable 8/8, legal: sum=60, ≤4/name, ≥1 Basic, ≤1 Ace Spec)

```
741×4 Abra                  [Basic Pokémon]
742×4 Kadabra               [Stage 1 Pokémon]
743×4 Alakazam              [Stage 2 Pokémon]
305×4 Dunsparce             [Basic Pokémon]
 66×3 Dudunsparce           [Stage 1 Pokémon]
1079×4 Rare Candy           [Item]
1086×4 Buddy-Buddy Poffin   [Item]
1152×4 Poké Pad             [Item]
1081×4 Enhanced Hammer      [Item]
1097×3 Night Stretcher      [Item]
1129×1 Sacred Ash           [Item]
1182×3 Boss's Orders        [Supporter]
1231×4 Dawn                 [Supporter]
1225×4 Hilda                [Supporter]
1184×1 Lana's Aid           [Supporter]
1264×1 Battle Cage          [Stadium]
  5×3 Basic {P} Energy      [Basic Energy]
 19×4 Telepath Psychic Energy [Special Energy]
 13×1 Enriching Energy      [Special Energy]
```
Method: own-side revealed-card aggregation (own active/bench/hand/discard for aidy's player index only), max-per-step
then MAX across 8 eps, cap ≤4 by name. No energy padding needed — raw max summed to exactly 60. Recon script:
scratchpad `aidy_decklist_recon.py`.

- [x] **F1-4 CUBCHOO PIVOT RECON (rejected on BC grounds):** Cubchoo/Water-box was evaluated as the
      pivot candidate from F1-2 (top-10 deck, ~30% of top-10). Recon 2026-07-03 (3 subagents; logs
      `/private/tmp/cubchoo_recon_decklist.txt`, `cubchoo_mechanic.txt`, `cubchoo_il_matchup.txt`).
      **VERDICT: COMBO/LOCK — BC will whiff. Do NOT pivot.** Same profile as Iono/Alakazam, NOT Crustle.
      Experts: MtN #14 (teamId 16408528, subId 54283045, μ1094.4, 235 eps), Michael Krager #13
      (teamId 16374997, subId 53900795, μ1096.6, 1000 eps), XP3RiX #7 (16384608, 54276772, μ1130.0, 300 eps),
      chamboabi (16428379, 54292983, μ1080.7, 108 eps — DIFFERENT archetype: Cubchoo+Walrein+Drednaw, not the lock list).
      Decklist (MtN, stable 8/8, legal sum=60): 506×4 Cubchoo, 117×1 Ogerpon ex (wall), 414×1 TR Articuno (wall),
      3×6 Basic {W}, 1166×3 Gravity Gemstone (retreat-lock tool), 1264×2 Battle Cage, 1247×1 Neutralization Zone,
      1256×2 TR Watchtower, 1182×4 Boss's Orders, 1219×4 TR Petrel, 1227×4 Lillie's Determination,
      1194×2 Colress's Tenacity, 1197×2 Xerosic's Machinations, 1086×2 Buddy-Buddy Poffin, 1120×4 Crushing Hammer,
      1081×4 Enhanced Hammer, 1119×3 Energy Search, 1097×3 Night Stretcher, 1152×2 Poké Pad, 1122×4 Pokégear 3.0,
      1137×1 Tool Scrapper, 1123×1 Switch. Beartic 507 NEVER used (0 copies) — pure Basic lock, no evolution.
      Win-con = **DECK-OUT** (100% of MtN's 5 wins by OPP_DECK_OUT, 0 prizes taken in 5/5 wins, avg 36.6 turns).
      Snotted Up (Cubchoo 506, {W} 10 dmg + "Defending can't attack next turn") IS the win-con, not disruption:
      lock → opp can't take prizes → opp burns deck → deck-out. 10 dmg incidental; 0 Articuno/Ogerpon attacks
      across 16 games. BC-fail reasons: (1) IL signal has NO prize-taking actions in winners — structurally
      opposite to Crustle prize-rush BC; (2) narrow corridor — must land Snotted Up every turn (one game = 32
      consecutive lock turns), a single dropped lock → Cubchoo HP70 dies → no recovery (10 dmg/turn can't catch
      up on prizes); (3) 3 losses show fragility (lock broke → lost fast, or lost mill race by recursion timing);
      (4) 30-65 turn precision horizon. Matchup (77 games, own-side archetype detection): Alakazam 19-8 (70%),
      Archaludon 10-0 (100%), Grimmsnarl 3-1 (small n), Okidogi 3-3, **Crustle 1-4 (20% — real counter, and
      Crustle is our champion + top-meta)**, Dragapult 1-1, mirror 1-1. IL volume: 410 winner-MAIN pairs locally
      (MtN+Krager), <2000 target — would need ~30-40 more eps harvested; ceiling ample (MtN 2937, Krager 12500).
      CONCLUSION: Cubchoo is a lock/mill deck BC cannot clone (wins by deck-out, not prize-rush; one missed lock
      turn loses), AND it loses to our own Crustle champion. Pivot REJECTED. Alakazam (F1-3) stays the IL target;
      if its gate also fails, the answer is a BC-friendly midrange deck (e.g. a Crustle-family beatdown), not a
      second lock deck. Reconsider Cubchoo only if pivoting from BC to a search/planner agent (PIMC/ExIt can plan
      the lock sequence; BC cannot imitate it).

- [x] **F1-5 GRIMMSNARL EX / MUNKIDORI TOOLBOX RECON:** top-3 meta deck (tonakaiiii #1 μ1247 wr0.50,
      kazuki0123 #3 μ1195 wr0.67 stab0.91). All attacks = Grimmsnarl ex "Shadow Bullet" (single universal
      attacker, direct dmg + 30 bench snipe); Munkidori "Adrena-Brain" is rare support tech (dmg-shifting),
      not the win-con. **VERDICT: MIDRANGE / BC-friendly — all attacks Shadow Bullet, Munkidori ability is
      support tech; top-1 BC target = clone kazuki0123 #3 (wr0.67 stab0.91) [tonakaiiii #1 wr0.50 as data
      supplement]**. Decklist (kazuki0123, 8/8 stable, legal sum=60): 648×3 Marnie's Grimmsnarl ex, 647×3
      Marnie's Morgrem, 646×3 Marnie's Impidimp, 66×2 Dudunsparce, 305×3 Dunsparce, 649×1 Marnie's Morpeko,
      112×4 Munkidori, 1159×1 Hero's Cape, 1086×4 Buddy-Buddy Poffin, 1079×3 Rare Candy, 1152×4 Poké Pad,
      1231×4 Dawn, 1227×4 Lillie's Determination, 1197×2 Xerosic's Machinations, 1097×2 Night Stretcher,
      1119×1 Energy Search, 1139×1 Energy Recycler, 1259×3 Spikemuth Gym, 7×12 Basic {D} Energy.
      IL volume (all local eps, winner-MAIN ctx=0): kazuki0123 793 pairs (25 eps), tonakaiiii 494 pairs
      (18 eps), combined 1287 — below 2000 target, needs more harvesting (both top-3, ceiling likely ample).
      Matchup (67 games, own-side marker detection, Grimmsnarl-side perspective): vs Alakazam 20-16 (56%),
      vs Archaludon 14-8 (64%), **vs Cubchoo 2-6 (25% — LOSES, confirms F1-4's Cubchoo-side 3-1 finding from
      the other direction)**, vs Crustle 1-0 (n=1, noise). Not a hard counter to the field, but a real hole
      vs Cubchoo.

## H — PLAN v4: HYBRID PIMC (PRIMARY path since 2026-07-11, user-confirmed)

> **Why:** потолок BC ≈800–930 подтверждён трижды (наш 930-пик; kazuki-клон 453 при эксперте 1083;
> PokéForge-клон ~802 = их лучший результат при провале всех собственных архитектур — форум 724187). Топ-100 = 989
> → BC не хватает. Единственный путь выше эксперта = поиск; топ-1 по таймингам использует тяжёлую модель + полный
> бюджет хода (форум 724362). §S провалился из-за конкретного дефекта (rollout обеих сторон = Crustle-IL →
> неверная оценка миров против не-Crustle, S4b −0.40), а не из-за «поиск не работает». v4 чинит opponent-model
> (H3-1) и добавляет чистые данные (H1) и калиброванную метрику (H2).

### H0 — Гигиена (0.5 дня)

- [ ] **H0-1 Git commit копилки** (не коммичено с 06-18: src/exit, src/pilot, src/cg_native, v010-v018, docs,
      data/engine, PLAN v3/v4). Потеря дерева = потеря всей PIMC-инфры и IL-моделей 3 архетипов. По команде
      человека: `git add -A && git commit -m "chore: checkpoint 07-03..07-11 (PIMC infra, native engine, v010-v018, plan v4)"`.
- [x] **H0-2 PLAN.md v4** — этот документ (2026-07-11).

### H1 — Аудит off-by-one в реплеях (~1-2 дня; БЛОКИРУЕТ retrain, НЕ блокирует H2)

- [x] **H1-1 Верификация гипотезы 717141** ✅ DONE 2026-07-11 (sonnet-субагент; `scripts/audit_replay_offbyone.py`,
      250 реплеев, сид 1234; отчёт `scratchpad/audit_offbyone_report.md`). **ВЕРДИКТ: NOT_SHIFTED для НАШЕГО
      пайплайна.** Форумный баг РЕАЛЕН, но в другом источнике: внутренний `visualize`-экспорт движка смещён
      (6301/6301 информативных шагов SHIFTED; `entry[0].selected`=null в 250/250 файлах), однако
      `src/replays/parse.py` его НЕ читает — наш источник = harness-пары `steps[i].observation` ↔
      `steps[i+1].action`: **0/7090** признаков сдвига (241 NOT_SHIFTED / 9 AMBIGUOUS / 0 SHIFTED из 250 файлов),
      кросс-сверка с `visualize.selected` совпадает на offset 0 (29005/29005; offset ±1 лишь 0.292).
      **IL-датасеты чисты; гипотеза «клон слаб из-за битых данных» ОПРОВЕРГНУТА — потолок BC настоящий.**
- [x] **H1-2 SKIPPED 2026-07-11 (ветка NOT_SHIFTED):** пересборка датасета/retrain НЕ нужны; prior остаётся
      текущим (`models/il_crustle.txt` / v009a); кандидата v019_crustle_fixed не будет — идём в H2/H3.
      Архив условия (если бы SHIFTED): исправить выравнивание в
      `src/replays/parse.py`, пересобрать Crustle-датасет с ТЕМ ЖЕ отбором, что у v007/v009a (winner-side,
      deck-matched), retrain рецептом C6 (lr=0.005, leaves=15, min_child=50, 300 деревьев, early-stop OFF) →
      `models/il_crustle_v4.txt`. ГЕЙТ: H2H новый-prior-агент vs v009a, 300g native, чередование сторон;
      PASS = WR≥0.55 И Wilson_lo>0.50. PASS ⇒ (i) prior для H3, (ii) кандидат **v019_crustle_fixed** на ранний
      live-probe (явное «ок» обязательно). NOT_SHIFTED ⇒ записать в память, prior остаётся, идти в H3.

### H2 — Анкерный оценочный пул + калибровка predicted-μ (~2-3 дня, параллельно H1)

> Локальный гейт vs свои агенты не предсказывает ладдер (v012 0.70→499; v018 66% vs Alakazam→484; PokéForge —
> гейт перепутал ранжирование). Чиним КАЛИБРОВКОЙ на якоря с известными live-скорами, а не отказом от локальной
> оценки.

- [x] **H2-1 `src/eval/anchored_pool.py` (Gauntlet-v3)** ✅ DONE 2026-07-11 (sonnet-субагент): режимы
      roundrobin/candidate/calibrate, resumable state, multiprocessing 4 воркера. Round-robin 21 пара × 300 игр
      = **6300 игр за ~2 мин (~50 игр/с)** — якоря без поиска летают; 0 exceptions/0 illegal, 9 ничьих.
      Матрица: `results/gates/anchored_pool_20260711.json`. **КРИТИЧЕСКИЙ ФИКС native-загрузки сабмитов:**
      v001/v012/v002 импортируют `cg.api` напрямую → их Linux-only `libcg.so` падал на macOS ARM; решение =
      алиас `sys.modules["cg"]→cg_native` (и подмодули) ДО exec их main.py (`api.py` побайтово идентичен).
      ОБЯЗАТЕЛЕН для всех будущих нативных харнессов, грузящих `submissions/*/main.py`.
- [x] **H2-2 Калибровка** ✅ DONE 2026-07-11 — **FAIL по критерию плана:** LOO Spearman **0.7143** (<0.8) и в
      варианте all-7 (s=575.9), и без v002 (s=549.5); цепочка нарушена (v009a предсказан выше v007; v012 завышен
      на ~+200 — тот же перекос «узкий контрщик Crustle», что жил в v012 0.70→499; пул структурно скошен к
      Crustle-семейству). ⇒ ПО ПРАВИЛУ ПЛАНА: **predicted-μ = только РАНЖИРОВЩИК кандидатов внутри пула, НЕ
      прогноз live; финальный арбитр — live-probe.** Follow-up DONE 2026-07-11: лучший вариант = **5 якорей БЕЗ v012 и БЕЗ v002**
      (`variant_no_v012_no_v002`, s=413.3): **Spearman 0.800**, инверсии только шумовые (v009a↔v007: Δμ̂=9.9 при
      live-Δ=9.1 — неразличимы) или вне контрольной цепочки (v015↔v001). РЕШЕНИЕ: это ранжировщик для H3-4(c);
      v012/v002 из калибровки исключены навсегда (как спарринги — можно); v009a+v007 = один «чемпионский блок»
      ~724-733. Вариант без v012 но С v002 — ХУЖЕ (0.543): v002=779.9 — вторая аномалия (июньский снимок,
      локально 11-15% vs Crustle-пара). `results/gates/anchored_calibration.json` (все 4 варианта).
- [x] **H2-3 Мета-деки экспортированы** ✅ DONE 2026-07-11: `decks/meta/{alakazam_aidy,grimmsnarl_kazuki,
      cubchoo_mtn}.csv` (по 60 строк, суммы копий сверены с §F1-3/F1-4/F1-5 построчно). Спарринг-пилоты
      v017/v018 запускаемы локально.

### H3 — PIMC v2: раздельные rollout-политики (~5-7 дней)

- [x] **H3-1 Opponent rollout policy ≠ свой prior (фикс root cause S4b):** для распознанного архетипа —
      дешёвый скриптовый пилот (v017-эвристика для Alakazam, v018-guards для Grimmsnarl, generic greedy-attack
      для прочих/неопознанных); своя сторона = Crustle-prior (v4 после H1) с ε-greedy ε≈0.3 (лит: Scopone
      ε∈[0.3,0.7]). LightGBM — только в корне/узлах, НЕ на каждом шаге rollout. ✅ DONE 2026-07-11 (smoke):
      `PIMCAgentV2` в `src/exit/pimc_agent.py`; `locked_archetype` детектится и фиксируется (mirror→crustle,
      alakazam→alakazam, grimmsnarl→grimmsnarl); `rollout_degradation={}` — ни одна политика не падала.
- [x] **H3-2 Root-parallel PIMC:** редетерминизация на КАЖДЫЙ rollout (сотни миров × 1 rollout, а не K миров ×
      глубокое дерево); multiprocessing по ядрам; замер ядер Kaggle: `os.cpu_count()` в лог validation-эпизода;
      бюджет 2-4 s/решение, bank-manager из S3 переиспользуется; поиск только на MAIN-развилках ≥2 нетривиальных
      опций (decision-gating из S3). ✅ DONE 2026-07-11 (smoke): rollouts/decision 38–96 при bank 0.5–3.0s;
      search_share 0.50–0.87 (decision-gating работает); bank_spent_max 52.4s < 90s — перерасхода нет.
- [x] **H3-3 Микро:** при выигрыше монетки выбирать ПЕРВЫЙ ход (+10пп, форум 723591). ✅ код в `PIMCAgentV2`
      (deck-list-choise). ⚠ НЕ валидирован smoke-форматом — движок `battle_start` не экспонирует deck-selection;
      отдельный тест первого хода нужен в H3-5 перед сабмитом.
- [x] **H3-0 SMOKE (работоспособность, НЕ гейт):** ✅ PASSED 2026-07-11. `scripts/pimc_v2_smoke.py` (3 матчапа
      × 4 партии = 12): **0 illegal, 0 exceptions**. Тайминги: ms/decision_max 1538ms < 3000ms budget;
      bank_spent_max 52.4s < 90s. W/L на 12 партиях НЕ значим (mirror 1-3, alakazam 1-3, grimmsnarl 4-0 —
      информационно). Артефакт: `scratchpad/h3_smoke.json`. cg-алиас sys.modules safety-net добавлен в smoke.
- [~] **H3-4 GATE Phase-1 (50g пилот, БАГОВАННЫЙ код): формально KILL_PIMC + RED_FLAG, НО аудит нашёл sort-баг →
      фикс применён → ре-гейт pending (см. Status Log (f)(g)).** Phase-1 (багован.) результат (481/700 игр):
      (a) mirror WR=**0.40** (14-21) — FAIL; (b) FIELD-дельта **4/5 оппонентов Δ<0** (v017 −0.171, v018 −0.030,
      meta:alakazam −0.143, meta:grimmsnarl −0.208, meta:cubchoo +0.223) — KILL; (c) μ̂=739<783 — FAIL;
      RED-FLAG WR vs v001_random=**0.727**. RED-FLAG оказался сигнатурой БАГА (а не «PIMC не работает»).
      **★ БАГ-АУДИТ (2026-07-11): root cause = SORT-БАГ в `src/pilot/pimc_guided.py:526`** —
      `avg_scores.sort(reverse=True)` (tuple (score,idx)) при равных score выбирал БОЛЬШИЙ idx (последний
      кандидат = END/CARD в порядке `_smart_candidates`: ATTACK,…,END,CARD) вместо меньшего (ATTACK). В зоне
      near-certain-win (49% решений vs random имели max score=1.0 → все кандидаты ~равны) sort-баг выбирал END
      вместо ATTACK → PIMC заканчивал ход → проигрыш random 27%; решал в 43% решений. Фикс (1 строка):
      `avg_scores.sort(key=lambda x: x[0], reverse=True)` — stable по score, на ties сохраняет порядок → ATTACK
      побеждает. **WR vs random 0.727 → 1.000** (20 игр, 2 seeds). Mirror 0.40→0.375 (n=8, шум — НЕ починен:
      архитектурное ограничение, PIMC над own prior не обыгрывает оппонента с той же стратегией; отдельная
      проблема от sort-бага). 6 гипотез аудита: H2 (rollout слабо-информативен в near-certain-win зоне) +
      H4 (opponent_model="auto" сэмплирует сильные meta-pilots даже для random — вторично) частично; H1/H3/H5/H6
      refuted кодом (C++ исходник движка подтвердил корректность перспективы/search-API/gating). Аудит-агент
      рекомендация: BUG_FOUND_AND_FIXED → ре-гейт H3-4 50g. Артефакты: `scratchpad/pimc_audit_log.json`,
      `scripts/pimc_bug_audit.py`, фикс в `src/pilot/pimc_guided.py:532`. ⚠ Тот же sort-баг в `pimc.py:460`/
      `pimc_v2.py:427`/`pimc_hybrid.py:350` — не активны для PIMCAgentV2 (preference: pimc_guided), опц. гигиена.
- [x] **H3-4-re GATE (ре-гейт после sort-фикса, 50g×14 ячеек = 700 игр, свежий state): KILL_PIMC ОКОНЧАТЕЛЬНО.**
      ✅ 2026-07-12, 0 illegal/0 exceptions. RED_FLAG **снят** (PIMC vs v001_random 0.92 — sort-фикс сработал).
      НО arm(a) mirror **0.34** (Wilson[0.22,0.48], хуже собственного prior'а) — FAIL; arm(b) FIELD-дельта
      **3/5 отрицательны**: v017_alakazam −0.12 (было −0.171), meta:alakazam −0.10 (−0.143), meta:grimmsnarl
      −0.02 (−0.208); только meta:cubchoo +0.12 и v018 0.00 — гейт требует Δ≥+0.05 на каждом плече, ни одного <0
      → FAIL/KILL. Sort-фикс подтянул ВСЕ дельты вверх (−0.171→−0.12), но **знак против alakazam-архетипа НЕ
      перевернулся** — остаточный вред глубже sort-бага (rollout неверно оценивает миры vs alakazam). Это
      эмпирически подтверждает recon [[project-search-vs-rl-ceiling]]: **search над сильной политикой
      (Crustle-IL ~906) не окупается**. **PIMC-ветка (§S + §H3) закрыта НАВСЕГДА.** Артефакты:
      `results/gates/h3_gate_re_state.json`, `h3_gate_re_phase1.json`; багованные `.buggy.bak.json`.
      Код PIMCAgentV2/pimc_guided НЕ удалять (история), но в сабмиты НЕ идёт.
- [ ] **H3-5 Packaging/timing:** native + Docker тайминги; НОВЫЙ `libcg.so` (ffd89bf9…) в бандле; presubmit
      exec-no-`__file__`; never-throw обёртка.

### H4 — Live-probe (финальный арбитр; ~2-3 дня/цикл)

- [ ] **H4-1** Каждый аплоад — ТОЛЬКО с явного «ок» пользователя. Порог: >733 (v007) за ~100-150 игр = ветка
      жива; ≥900 = успех; <700 = откат (ре-сабмит лучшего проверенного) + диагностика по реплеям.

### H5 — Компаундинг (опционально, после успеха H4)

- [ ] **H5-1 ExIt-lite:** 1-2 раунда дистилляции PIMC-траекторий в LightGBM-prior (НЕ полная петля с нуля).
- [ ] **H5-2 Deck-swap:** PIMC деко-агностичен → примерить Dragapult/Dusknoir или Alakazam+Dudunsparce;
      решение по H2-метрике + свежему мини-рекону меты.
- [ ] **H5-3 Еженедельный мини-рекон** (LB, форум, эпизоды) — мета ротирует ~каждые 2 недели.

### H6 — Freeze (08-09..08-16)

- [ ] **H6-1** К 08-09 выбрать финальную пару; с 08-11 — НИКАКИХ экспериментов (существующее правило);
      tracked-2 = чемпион + лучший проверенный вариант.

---

## Отвергнутые пути + почему (НЕ ходить по кругу)

| Путь | Доказательство провала |
|---|---|
| BC на комбо/лок деках (Iono, Alakazam, Cubchoo) | live 336–507; гейт 43%; Cubchoo выигрывает дек-аутом — BC структурно мимо |
| Клон новой топ-деки как обход потолка BC | v018 453–484 при эксперте kazuki 1083 (~40% силы); PokéForge-клон 802 = их потолок |
| Наращивание ОБЪЁМА IL-данных | v008: 3× данных → mirror 0.394, регресс |
| Каунтер-гейт vs свой чемпион как критерий сабмита | v012: 0.70 локально → 499.2 live |
| Ручные патчи поверх сильной политики без поиска | PokéForge: 0 измеримого прироста; наши v012→v013 (38%) |
| 1-ply / rollout-search ≈ своя же политика | v003: три гейта ≈0.50 (06-17) |
| PIMC с rollout ОБЕИХ сторон = свой prior | S4b: −0.40 vs Alakazam, −0.15 vs Grimmsnarl (чинится H3-1; как было — не повторять) |
| Полный ExIt/AlphaZero-цикл с нуля | PokéForge self-play/DAgger/A2C провал; ~36 дней до дедлайна; только ExIt-lite поверх рабочего PIMC |
| IS-MCTS как первая реализация | детерминизацию всё равно требует API движка; merge деревьев между процессами дорог (лит-рекон 07-11) |
| Ре-сабмит как «сохранение» рейтинга | рейтинг не наследуется (форум 714189); v007-resub карабкался 600→733 заново |
| Микротюнинг деклиста под замороженный пилот | Phase-3 гаунтлет: все recount-варианты статистически неотличимы от базы |

## S — SEARCH PIVOT: PIMC on Crustle (CLOSED 2026-07-05: S4a≈0.50, S4b −0.40/−0.15 → superseded by §H)

> **Why:** BC/IL is capped ~930 and decayed to 760; top-1 = 1223. Imitation can't exceed the expert. The engine's
> own search API + native engine (4200 rollouts/turn, budget 99.7% unused) is the only lever with headroom above 930.
> **Thesis to test:** flat IL plays ONE fixed policy vs the whole field; PIMC re-plans per detected opponent by
> determinizing the hidden state against the actual opp archetype (Alakazam 33% / Archaludon 22% / Grimmsnarl /
> Cubchoo) and rolling out with the IL prior. The headroom = per-opponent adaptation + look-ahead past IL's myopia.
> **Build on Crustle** (our only pilot that works = strong rollout prior). NO new decks — deck-hopping burned us 3×.
>
> **Algorithm (Perfect-Information Monte Carlo):** at a high-stakes MAIN fork with options {a}: for each a, repeat K
> determinizations — sample a plausible hidden state (opp deck from meta prior + opp hand/prizes consistent with
> revealed cards), apply a, roll out to terminal with the Crustle IL as the policy (both sides), record result
> (`obs.current.result`); pick argmax mean win. Cheap/forced contexts (single legal, promotion/energy guards) skip
> search and use the existing IL + deterministic guards.

- [x] **S1 Re-parameterize `src/exit/pimc_agent.py`** ✅ DONE 2026-07-05. Smoke `scripts/pimc_crustle_smoke.py`
      (`.venv/bin/python scripts/pimc_crustle_smoke.py`): 3/3 runs, 0 exceptions, 0 illegal moves; PIMC drives
      53–72% of decisions (search), prior 12%, trivial 16–35%. `crustle_prior` = dynamic load of champion
      `submissions/v009a_crustle/main.py` `agent()` (1:1, no reimpl). **Timing native ARM budget=0.3s: median
      ~150ms/decision, max ~320ms**, ~9–13 worlds/decision → **~30× headroom** (600s bank / ~120 dec ≈ 5s/dec avail).
      Bug fixed: `src/pilot/pimc_guided.py` hard `from cg.api` crashed on macOS ARM → try `cg_native.api` first.
- [x] **S2 Determinizer** ✅ DONE 2026-07-05 (`src/exit/pimc_agent.py`). Added reconstructed `ALAKAZAM_DECK`/
      `GRIMMSNARL_DECK`/`CUBCHOO_DECK` (+ verified `CRUSTLE_DECK_APPROX` vs deck.csv), all `assert len==60`;
      archetype signatures `{741,742,743}`/`{646,647,648}`/`{506}`/`{344,345}` → `identify_archetype()`;
      `META_PRIORS` (ala .33/arch .22/cub .07/grimm .07/crus .05/generic .26) + `sample_meta_deck()` fed PER-WORLD
      so K determinizations spread across the field; persistent `self._revealed_opp_ids`+`self._locked_archetype`
      accumulate across the game and lock once a signature is unambiguous. `opponent_model` now
      mirror|crustle|auto|meta. Unit smoke `scripts/pimc_determinizer_smoke.py` (detection + ±0.035 prior match +
      side-isolation) OK; S1 smoke still 0 exc/0 illegal; timing unchanged. **TODO: Archaludon (~22%) has NO
      reconstructed list → mapped to Crustle placeholder; reconstruct from `data/episodes/all_teams` if S4 shows bias.**
- [ ] **S3 Bank manager + decision-gating:** budget ~2–4 s/decision, K determinizations × rollouts, self-limit to
      keep the 600s bank safe over a long game; only search MAIN forks with ≥2 non-trivial options. Verify timing
      NATIVELY (`src/cg_native/`, ×2.5 faster) AND in Docker (submission runtime = linux x86_64). Multiprocessing
      only if parallelizing — threading is SIGSEGV/corruption-unsafe (in-submission search is single-process seq.).
- [ ] **S4 GATE — the real test (two arms):** (a) **H2H vs its own prior:** PIMC(Crustle) vs v009a flat IL, 300g,
      side-alternating — MUST win (WR≥0.52, Wilson_lo>0.50). A search that can't beat its own rollout policy is the
      old 1-ply failure repeating → stop and rethink K/depth/determinization. (b) **Gauntlet-v2 vs reconstructed meta
      decks** {Alakazam(aidy list §F1-3), Grimmsnarl(kazuki list §F1-5), v007, v009a, v002 Lucario, random} — where
      flat IL's ceiling shows; PASS = ≥55% weighted, no matchup <45%. Bundle the NEW `libcg.so` for these sims.
- [ ] **S5 Live probe (2–3 days, ~100–150 games) — final arbiter.** Explicit human approval per upload. Bundle NEW
      `libcg.so`, presubmit exec-no-`__file__`, Docker timing check. Success = beats v007's 760 (ideally >930). Fail
      the H2H (S4a) → the search adds nothing → fall back to F-plan (polish IL) or escalate to ExIt (F3/F4).
- [ ] **S6 If S4a passes but gains are thin:** the bottleneck is the rollout policy quality → this naturally becomes
      the first ExIt iteration (PIMC(IL) generates better-than-IL trajectories → retrain IL on them → stronger prior
      → repeat). That is the F4 loop, finally with a real reason to run it.

## F2 — (SUPERSEDED by §S for now) Pilot v2 for a NEW deck (by ~07-13)

> Kept for reference. New-deck BC is deprioritized after 4 BC failures; revisit only if §S search fails AND a
> genuinely BC-friendly midrange target appears. The Crustle IL we already have is the §S rollout prior.

- [ ] **F2-1 Feature module** for the chosen archetype (66-dim skeleton transfers; `src/il/features_v011.py` is the
      template).
- [ ] **F2-2 IL** via existing `src/exit/pilot.py` `Pilot.train_from_replays` (deck-agnostic, `feature_fn`
      parameterizable). Offline top-1 = checkpoint only, NOT a decision.
- [ ] **F2-3 GAUNTLET-V2 GATE (new standard, replaces single-champion gate for new-deck candidates):** opponent
      pool = {v007_crustle, v009a_crustle, lucario heuristic (v002), reconstructed top-meta decks, random}.
      PASS = ≥55% weighted overall AND no key matchup <45% (anti-v012 rule: no narrow counters) AND presubmit
      exec-no-`__file__` AND timing safe. Same-deck iterations may still use champion H2H (300g, Wilson_lo>0.50).
- [ ] **F2-4 Live probe 2-3 days (~100-150 games)** — the real metric. Every upload needs explicit human approval.
      Fail → re-submit champions (slots allow).

## F3 — PIMC on top of pilot + first-ever ExIt run (by ~07-20)

- [ ] **F3-1 Parameterize PIMC** (`src/exit/pimc_agent.py`): rollout policy = pilot v2 via the existing `prior_agent`
      hook (currently HARD-WIRED to the dead Iono heuristic — audit 07-03); opponent model = F1 meta priors +
      revealed-cards detect (skeleton exists).
- [ ] **F3-2 Budget 2-4 s/decision + bank manager** (600s bank; verify native + Docker timings).
- [ ] **F3-3 First-ever smoke of `src/exit/exit_loop.py`** (`./scripts/run_exit.sh` 1 iter / 20 games) — the ExIt
      loop has NEVER been executed (audit 07-03); shake out integration bugs.
- [ ] **F3-4 Gauntlet-v2 gate + live probe.** Expectation: +5-15pp over pure IL.

## F4 — ExIt compounding (07-20..08-08)

- [ ] **F4-1 Full iterations:** PIMC(pilot) vs frozen pool → winner-filter → retrain → gate → promote.
      Volume natively on macOS and/or Kaggle notebooks.
- [ ] **F4-2 Weekly mini meta-recon;** re-target the pipeline if the meta shifts (pipeline is deck-agnostic).
- [ ] **F4-3 Optional:** value model to truncate rollouts if full-game rollouts bottleneck.

## F5 — Freeze & final pair (08-09..08-16)

- [ ] **F5-1 By 08-09 choose the final pair** (champion + best vetted variant; possibly 2 archetypes as a meta
      hedge — decide by data).
- [ ] **F5-2 From 08-11: NO experiments** (existing freeze rule). Ensure tracked-2 = final pair; monitor
      post-deadline games.

---

## Decision rules for different outcomes (the "what if")

| Situation | Action |
|---|---|
| Candidate fails the gate (<55% or CI not clear) | Do NOT submit. Iterate or discard. Keep champion. |
| Candidate passes gate but ladder μ drops after ≥10 episodes | Roll back: re-submit champion. Diagnose via episode replays. |
| Engine per-move time limit (Q-TIME) is tight | Lower search K/depth; precompute; fall back to heuristic. Profile natively (F0-2b). |
| `torch` not importable in submission runtime (Q-DEPS) | Use LightGBM / numpy-only inference; export model accordingly. |
| Docker on ARM too slow for big gauntlets | Run volume self-play natively (F0-2b) or on Kaggle notebook; Docker for final validation. |
| Meta shifts (new strong opponents in replays) | Re-run F1 recon; re-target the deck-agnostic pipeline (F4-2). |
| Out of submission budget for the day | Stop submitting; keep iterating locally; resume tomorrow. |
| Within 5 days of deadline (≤2026-08-11) | FREEZE experiments. Ensure last 2 = champion + best vetted variant only. |
| A phase stalls / is too costly for the gain | Skip forward; ML/RL is the ambition but never at the cost of a stable last-2. |

---

## Status Log (append-only; newest on top)

- **2026-07-12 (k)** — **★ DECK-TECH: v023_crustle_techD закрывает обе дыры, паритет по полю → кандидат на live-пробу.**
  Deck-tech аудит (sonnet) + разбавленный вариант (оркестратор). Механика дыр: Alakazam Powerful Hand (743, dmg=0,
  кладёт 2 counter/карту в руке) — **damage-counters обходят ЛЮБОЙ prevent-damage щит движка** (не только Rock Inn,
  подтв. кодом v016), лечится ТОЛЬКО hand-disruption; Cubchoo = control (Crushing/Enhanced Hammer + Boss + Team
  Rocket's Watchtower снимает Rock Inn симметрично). techA (агент: +Judge×3+Xerosic×3+Payapa×2, снял весь heal) —
  Alakazam +0.11 но просадка FestivalLead/MegaVenusaur (survivorship trap: убрал Jumbo Ice Cream heal). **techD
  (разбавленный: +Judge×3 +Xerosic×1, снял только Jumbo Ice Cream×2/Cook×1/Waitress×1, heal сохранён):** полный
  no-regression 60g×19 дек — alakazam **+0.167** (0.35→0.52), cubchoo **+0.183** (0.63→0.82), обе дыры закрыты; 7
  лёгких матчапов просели −0.04…−0.07 (heal-consistency переложена в дыры); **overall 0.9118→0.9106 = паритет**.
  По ожидаемому рейтингу на СМЕЩЁННОМ поле (recon: Alakazam>1100, Starmie/Lucario топ) techD должен обгонять baseline
  сильнее. v023 грузится Kaggle-стиль (exec без __file__) OK, 60 карт. Kill-риск лёгких матчапов на 60g может быть
  шумом. Slot свободен (0/5 сегодня). Артефакты: `decks/tech/crustle_{A,B,C,D}.csv`, `submissions/v020-v023`,
  `results/gates/field_full_report.json`. NEXT = решение о live-пробе (у пользователя). [[project-meta]] [[project-plan-v4]]
- **2026-07-12 (j)** — **★ ДЕК-VIABILITY: Crustle = лучшая наша дека, проблема = 2 контр-архетипа.** Новый
  `scripts/field_viability.py` (наш пилот+дека vs 19 top-cut мета-дек из recon, generic greedy-пилот, seat-swap):
  2280 игр за 55с, 0 illegal/0 exceptions. **v009a_crustle WR=0.926** (Wilson[0.90,0.94], 672-54) — доминирует;
  v018_grimmsnarl 0.71, v017_alakazam 0.65 (обе хуже держат поле). Crustle бьёт 17/19 архетипов на 0.90-1.00, НО
  **проваливает два: alakazam_aidy 0.55 (22-18), cubchoo_mtn 0.60 (24-16)** — оба обходят «Mysterious Rock Inn»
  (щит только vs ex-атак; Alakazam damage-counter placement + Cubchoo не-ex атаки проходят сквозь, как Iono). И это
  против СЛАБОГО greedy-пилота — реальные IL-пилоты этих дек на ладдере дожимают → объясняет дрейф 930→733.
  **ВЫВОД: deck-swap НЕ нужен (Crustle сильнейшая), вектор = закрыть Alakazam+Cubchoo matchup.** ОГОВОРКА:
  greedy-пилот слабее IL → абсолютные WR завышены, но относительный скрин между нашими деками валиден. Артефакты:
  `results/gates/field_viability_report.json`, `decks/meta/*.csv` (19 дек). NEXT = как чинить 2 матчапа (tech-карты
  в деклист / matchup-specific пилот-правила / IL+PPO на этих матчапах). [[project-plan-v4]] [[project-meta]]
- **2026-07-12 (i)** — **★ H3-4-re KILL_PIMC ОКОНЧАТЕЛЬНО (честный, не багованный).** Ре-гейт после sort-фикса,
  700 игр (9 PIMC-ячеек × 50g + 5 v009a-ячеек), 0 illegal/0 exceptions. **RED_FLAG снят** (PIMC vs random
  0.727→0.92 — sort-фикс подтвердился). НО mirror **0.34** (хуже prior'а), arm(b) FIELD-дельта **3/5 отрицательны**
  (v017_alakazam −0.12, meta:alakazam −0.10, meta:grimmsnarl −0.02; meta:cubchoo +0.12, v018 0.00). Sort-фикс
  подтянул все дельты вверх (напр. v017 −0.171→−0.12), но **знак vs alakazam-архетипа не перевернулся** —
  остаточный вред глубже sort-бага. Эмпирически подтверждает recon: **search над сильной политикой не окупается**
  ([[project-search-vs-rl-ceiling]]: beam −15.4pp на сильной Starmie). **PIMC (§S+§H3) закрыт НАВСЕГДА.** μ̂-якоря:
  vs v015 0.66, v001 0.92, v018 1.00, v009a 0.34, v007 0.64. Гейт-процесс остановлен. **NEXT = новый вектор:**
  дек-viability vs расширенное мета-поле → улучшение политики (ExIt-lite дёшево, потом PPO +65 ELO). Развилка у
  пользователя. Артефакты `results/gates/h3_gate_re_*.json`. [[project-plan-v4]]
- **2026-07-11 (h)** — **RECON: +357 эпизодов, 169 топик-дампов, дайджест `docs/kaggle_recon_2026-07-11.md`.**
  Workflow `wf_c05e4860` (95/153 агентов ОК; 58 упали на session-429 субагентов — я перефанаутил, агент на КАЖДЫЙ
  из ~50 топиков вкл. старые 709xxx, 2.46M токенов выбили лимит, который затем убил ре-гейт H3-4-re). Синтез сделан
  оркестратором вручную из journal.jsonl. **★ Стратегический сдвиг:** независимые команды подтверждают **search
  ОБРАТНО ∝ качеству базовой политики** (beam +11.3pp на слабой Alakazam / **−15.4pp на сильной Starmie**) → наш
  Crustle-IL сильный (~906) → PIMC mirror 0.40→0.375 = **не только sort-баг, а фундаментальный потолок**. Единственный
  ИЗМЕРЕННЫЙ путь наверх поверх чемпиона = **Crustle IL + PPO = +65 ELO** (KL-penalty 0.01-0.09 к frozen BC prior).
  Card-embedding пробивает BC feature-потолок (68.5→74%). Методология: N≥1200+≥0.5pp пороги (наш 50g пилот мал для
  тонких дельт); forensic loss-analysis = трап (−7.6pp survivorship) → против планового ExIt-lite. Мета сместилась #3
  (Starmie/Lucario 42%/Alakazam>1100/Archaludon/Festival Lead; Iono контрит Crustle 0-10; поле ~3550). Незакрыто:
  439 эпизодов + ~40 старых топиков + парсинг 389 реплеев (можно локально через parse.py без субагентов).
  [[project-search-vs-rl-ceiling]] [[project-meta]]
- **2026-07-11 (g)** — **★ БАГ-АУДИТ PIMC: root cause НАЙДЕН, фикс применён, ре-гейт pending.** Пользователь выбрал
  вектор B (баг-аудит). Sonnet-агент: 20 игр vs random (instrumentation 413 search-решений) + сверка с C++ исходником
  движка. 6 гипотез: **root cause = SORT-БАГ** `src/pilot/pimc_guided.py:526` (`avg_scores.sort(reverse=True)` на
  tuple (score,idx) → при равных score выбирает БОЛЬШИЙ idx = END/CARD вместо ATTACK). В зоне near-certain-win
  (49% решений vs random max score=1.0) → PIMC выбирал END вместо ATTACK → проигрыш random 27%; решал в 43%
  решений, decisive в 4 проигрышах. Фикс 1 строка: `avg_scores.sort(key=lambda x: x[0], reverse=True)` (stable по
  score, ties→исходный порядок→ATTACK). **WR vs random 0.727→1.000** (20 игр, 2 seeds). Mirror 0.40→0.375 (n=8 шум,
  НЕ починен — архитектурное: PIMC над own prior не обыгрывает ту же стратегию). H1 (seat)/H3 (candidates)/H5
  (search-API)/H6 (gating) refuted кодом (C++ исходник: yourIndex=selectPlayer корректен, search_begin/step/end
  корректны, Search.clear() освобождает). H2 (rollout слабо-информативен в near-certain-win) + H4
  (opponent_model="auto" сэмплирует сильные meta-pilots даже для random) — частично, вторично. PIMCAgentV2 делегирует
  `_evaluate_candidates` в pimc_guided (не патчит сортировку) → фикс в боевом пути. Артефакты:
  `scratchpad/pimc_audit_log.json`, `scripts/pimc_bug_audit.py`, фикс `src/pilot/pimc_guided.py:532`. Тот же баг в
  `pimc.py:460`/`pimc_v2.py:427`/`pimc_hybrid.py:350` (не активны для PIMCAgentV2, опц. гигиена). **NEXT = §H3-4-re
  ре-гейт 50g СВЕЖИЙ state** (старые 481 игра — багован. код). Ожидание: RED_FLAG снят; arm (b) FIELD-дельта должна
  улучшиться (sort-баг вредил vs реальных оппонентов тоже); arm (a) mirror может остаться <0.55 (архитектурное).
  [[project-plan-v4]]
- **2026-07-11 (f)** — **★ H3-4 Phase-1 KILL_PIMC + RED_FLAG_STOP — НО с сигнатурой БАГА.** 50g пилот (481/700
  игр, вердикт стабилен с n≈20; прогон остановлен на ~70% т.к. результат однозначен). `scripts/h3_gate.py`
  (multiprocessing spawn 4 воркера, PIMCAgentV2 создаётся в воркере, BT-фит μ̂), bank_total_sec=600. 0 illegal,
  0 exceptions, 0 rollout_degradation по всем 481 играм — инфраструктура работает. Результат:
  (a) mirror WR=**0.40** (PIMC проигрывает своему prior'у v009a);
  (b) FIELD-дельта **4/5 оппонентов Δ<0**: v017 −0.171, v018 −0.030, meta:alakazam_aidy −0.143,
  meta:grimmsnarl_kazuki −0.208, meta:cubchoo_mtn +0.223 (единственный плюс) → KILL-критерий (b) сработал;
  (c) μ̂=739.0 < 783 (s=413.3) — FAIL; **RED-FLAG: WR vs v001_random = 0.727 < 0.90**. Формально KILL зафиксирован.
  **НО: проигрыш random 27% — аномалия.** Корректный PIMC поверх сильного prior'а (v009a бьёт random ~95%+)
  не должен опускаться ниже базовой политики; «стратегия-fusion» снижает качество относительно oracle, но не
  ниже собственной rollout-политики. Активный вред (0.40 mirror, 0.727 vs random) = сигнатура бага в
  (кандидаты|оценка-rollout|детерминизация|search-API|decision-gating), а НЕ доказательство «PIMC не работает для
  этой меты». План-ный KILL-критерий (b) этого не различал. Развилка передана пользователю:
  (A) честь KILL → fallback (deck-swap BC / re-examine BC ceiling); (B) баг-аудит PIMC 1-2 дня → фикс → ре-гейт,
  и если бага нет — подтверждение KILL; (C) pivot RL/self-play (топ-1 = тяжёлая модель+тайм-бюджет, форум 724362);
  (D) принять потолок BC ~800-930, оптимизировать Crustle/re-submit. Артефакты: `results/gates/h3_gate_state.json`,
  `results/gates/h3_gate_phase1.json`, `scripts/h3_gate.py`. [[project-plan-v4]]
- **2026-07-11 (e)** — **H3 SMOKE PASSED (работоспособность; НЕ гейт силы).** `scripts/pimc_v2_smoke.py`:
  3 матчапа × 4 партии = 12 игр, нативный движок macOS arm64. **0 illegal, 0 exceptions.** Код PIMC v2
  (`PIMCAgentV2` в `src/exit/pimc_agent.py`, ~92 КБ — туда ушла вся логика, отдельного pimc_v2.py нет):
  H3-1 раздельные rollout-политики работают (`locked_archetype` стабилен: mirror→crustle, alakazam→alakazam,
  grimmsnarl→grimmsnarl; `rollout_degradation={}`); H3-2 редетерминизация + bank-manager (rollouts/decision
  38–96, search_share 0.50–0.87, ms/dec_max 1538ms < 3.0s budget, bank_spent_max 52.4s < 90s — перерасхода нет);
  H3-3 первый ход закодирован, но НЕ валидирован smoke-форматом (`battle_start` не экспонирует deck-selection
  → отдельный тест в H3-5). W/L на 12 партиях НЕ значим (mirror 1-3, alakazam 1-3, grimmsnarl 4-0) — это smoke
  на 0-exc/0-illegal, не оценка силы. cg-алиас `sys.modules["cg"]→cg_native` safety-net добавлен в smoke-скрипт
  (оппоненты v009a/v017/v018 cg на верхнем уровне не импортируют, но safety на будущее). Артефакт:
  `scratchpad/h3_smoke.json`. **NEXT = H3-4 полный гейт** (300g mirror + FIELD-дельта + predicted-μ>783;
  KILL-критерий: (b) снова отрицательный → PIMC закрыт, fallback v019+deck-swap BC). Замечания на мониторинг
  в H3-4: (1) grimmsnarl длинные партии 131–204 ходов → per-decision гистограмма вместо median/max; (2)
  rollouts/decision на alakazam ниже (38.5 vs 96.2 mirror) — проверить не падает ли качество поиска;
  (3) auto-детект перебирает все 4 политики перед локом на первых решениях (overhead, некритично).
  [[project-plan-v4]]
- **2026-07-11 (d)** — **H2 follow-up: калибровка дожата, ранжировщик выбран.** Пересчёт на тех же 6300 играх:
  без v012 (6 якорей) = Spearman 0.543 (ХУЖЕ — искажение переходит на v002); **без v012 И без v002 (5 якорей,
  s=413.3) = Spearman 0.800**, обе инверсии шумовые/вне цепочки. РЕШЕНИЕ: гейт H3-4(c) использует
  `variant_no_v012_no_v002` как РАНЖИРОВЩИК (не абсолютный прогноз); v012 (узкий контрщик, +200 перекос) и v002
  (устаревший снимок 779.9, локально 11-15% vs Crustle-пара) исключены из калибровки навсегда; v009a+v007 =
  один блок ~724-733; RED-FLAG-правило: кандидат <90% vs v001_random ⇒ стоп. Все 4 варианта в
  `results/gates/anchored_calibration.json`. NEXT = H3-1/H3-2 (сборка PIMC v2). [[project-plan-v4]]

- **2026-07-11 (c)** — **H2 DONE (sonnet-субагент): Gauntlet-v3 построен; калибровка = FAIL критерия плана.**
  Round-robin 7 якорей: 21 пара × 300 игр = 6300 игр за ~2 мин (~50 игр/с, 4 воркера, 0 exc/0 illegal) —
  скорость на 1-2 порядка выше плановой оценки (якоря без поиска). Калибровка: LOO Spearman **0.7143 < 0.8**
  в обоих вариантах (all-7 s=575.9; без v002 s=549.5); v012 завышен на ~+200 (перекос «контрщик Crustle» —
  пул скошен к Crustle-семейству), v009a↔v007 инвертированы. ⇒ predicted-μ = ранжировщик внутри пула, НЕ
  прогноз live (правило плана сработало); live-probe остаётся арбитром. Запрошен follow-up: калибровка без
  v012-якоря. ВАЖНЫЙ ИНЖЕНЕРНЫЙ ФАКТ: нативная загрузка v001/v012/v002 требует алиаса
  `sys.modules["cg"]→cg_native` до exec main.py (их бандловый Linux libcg.so падает на macOS). H2-3: мета-деки
  экспортированы в `decks/meta/*.csv`. Артефакты: `src/eval/anchored_pool.py`,
  `results/gates/anchored_pool_20260711.json`, `anchored_calibration.json`. NEXT = H3-1. [[project-plan-v4]]

- **2026-07-11 (b)** — **H1 DONE (sonnet-субагент): off-by-one аудит = NOT_SHIFTED для нашего пайплайна.**
  Баг 717141 подтверждён, но только во внутреннем `visualize`-экспорте движка (6301/6301 информативных шагов
  смещены; `entry[0].selected`=null 250/250) — `src/replays/parse.py` его не использует. Наш источник —
  harness-пары `steps[i].observation`↔`steps[i+1].action`: 0/7090 признаков сдвига, 0 SHIFTED-файлов из 250,
  кросс-сверка с `visualize.selected` = точное совпадение на offset 0 (29005/29005). ВЫВОДЫ: (1) IL-датасеты
  чисты, H1-2 retrain отменён, v019 не будет; (2) гипотеза «BC слаб из-за битых данных» опровергнута — потолок
  BC ~800-930 настоящий, что УСИЛИВАЕТ ставку v4 на поиск (H3). Артефакты: `scripts/audit_replay_offbyone.py`,
  `scratchpad/audit_offbyone_report.md`. NEXT = H2-1 (анкерный пул). [[project-plan-v4]]

- **2026-07-11** — **PLAN v4 ADOPTED: HYBRID PIMC (вектор A, подтверждён пользователем через AskUserQuestion).**
  Полный рекон 3 субагентами. (LIVE) tracked-2={v018 484.1 (1W-6L), v007 733.1}, team #1790/4773; top-1 Yushin
  Ito 1274–1346 (волатилен), top-10 1101.6, top-100 989.0, поле +16%; неизменный v007 930.7→733.1 — рейтинг
  дрейфует с метой; ре-сабмит не наследует рейтинг. (ФОРУМ) 717141: возможный off-by-one в `selected`
  реплей-экспорта (организатор молчит) — все IL-пары под подозрением; 724187 PokéForge-постмортем: клон публичной
  политики ~802 = их лучший результат, self-play/DAgger/planning провалились, их оффлайн-гейт перепутал
  live-ранжирование; 724362: топ-1 = тяжёлая модель + полный тайм-бюджет (поиск в бою); 723591: первый ход +10пп.
  (ЛИТ) root-parallel PIMC с редетерминизацией на rollout — стандарт; ε-greedy rollout ε≈0.3-0.5; LightGBM в
  узлах, не в шагах rollout; ISMCTS не окупается при нашем API; ExIt только lite; методика Kaggle-симов =
  якорение локального пула на live-скоры своих сабмитов. (РЕПО) S4a/S4b негативы перепроверены по
  `scratchpad/s4b_field.json`; «Gauntlet-v2» как скрипт никогда не существовал; в оценочном пуле только свои
  агенты; git не коммичен с 06-18. РЕШЕНИЕ: PLAN.md переписан → §H0-H6 (аудит → анкерный пул → PIMC v2 с
  раздельными rollout-политиками → live-probe → опц. ExIt-lite/deck-swap) + раздел «Отвергнутые пути».
  NEXT = H0-1 (git commit) / H1-1 (аудит). [[project-plan-v4]]

- **2026-07-05 (c)** — **S4b FIELD test = DECISIVE NEGATIVE → SEARCH BRANCH CLOSED.** PIMC-Crustle vs flat
  IL-Crustle, each vs real field agents (N=20/arm/opp, native, `scripts/pimc_field_gauntlet.py`,
  `scratchpad/s4b_field.json`): **vs Alakazam(v017) PIMC 0.15 vs IL 0.55 (delta −0.40, clean/no-ceiling); vs
  Grimmsnarl(v018) PIMC 0.80 vs IL 0.95 (delta −0.15).** Both negative → search ACTIVELY HURTS vs the field.
  ROOT CAUSE (fundamental): PIMC's rollout policy for BOTH sides is the Crustle IL → it evaluates the opponent's
  moves as if they play Crustle-style; real Alakazam(combo)/Grimmsnarl differ → world-values systematically wrong
  → picks worse than the confident flat IL (worse the more the opp differs from Crustle). **This chains to the SAME
  wall as BC:** PIMC needs accurate per-archetype opponent POLICIES; building those = BC on combo decks = the thing
  that fails. Circular. 0 illegal/0 exc, bank-safe (max 200/600s). **VERDICT: search (S1–S6) is DEAD in this
  implementation; per the agreed criterion (both S4a mirror≈0.50 AND S4b field<0) the branch is closed. No more
  search experiments.** → consolidate on the proven Crustle IL (retune for current meta) — see plan v4 decision.
  [[project-search-pivot]]

- **2026-07-05 (b)** — **S1–S3 built, S4a H2H = NEGATIVE (clean).** PIMC(Crustle, IL prior) vs flat v009a IL,
  Crustle mirror, native: first run 0.41 but CONFOUNDED by my setup bug (`opponent_model="auto"` → PIMC modeled
  its mirror opp as the meta-FIELD, not Crustle). Fixed → `opponent_model="crustle"`: **PIMC ≈ 0.50 (10-10 @20g
  before a background-runner kill).** Instrumented rollout-prior degradation = **0.0% (0/89012 exc)** → the rollout
  policy IS the IL, clean; the 0.50 is a FAIR test. Deep search (13-40 searches/game, ~1.4s/decision, bank-safe
  45/300s), 0 illegal/0 exc. **VERDICT: deep PIMC over the Crustle IL does NOT beat the flat IL in the mirror** —
  4th time a search approach lands ~0.50 vs its base policy (P2 1-ply ×2 + this). CAVEAT: mirror is the WEAKEST
  case for PIMC (no opp-archetype uncertainty to exploit); the real thesis (per-opponent adaptation vs the FIELD)
  is untested BUT unmeasurable locally (no strong Alakazam/Grimmsnarl opponent agent). **This ALSO undercuts ExIt**
  (needs PIMC(IL)>IL to compound — measured ≈). Note: engine is non-deterministic (`random_device`) → same seed
  ≠ same game. Artifacts: `scripts/pimc_vs_il_h2h.py`, logs in scratchpad. Awaiting user strategic call. [[project-search-pivot]]

- **2026-07-05** — **STRATEGY PIVOT v3 → SEARCH (user-confirmed via AskUserQuestion).** Live-probe verdicts in:
  v018_grimmsnarl = **453.5** (below random) — BC failed even on the "BC-friendly" Grimmsnarl; kazuki0123 (cloned
  expert) LIVES 1083 vs our 453 clone = ~40% skill capture. v007=760.4, v009a=724.1 (decayed from 930/905),
  v009b(old)=910.2. Top-1 Majkel1337=1223. Web recon: public SOTA rule-based (repo `wmh/ptcg-abc`) = 836 (Bellibolt);
  their finding "simple deck cleanly piloted > complex combo, real ladder is the only judge" = our lessons exactly;
  our Crustle IL (930) already beat public knowledge. CONCLUSION: BC/IL is capped ~800–930 (imitation ≤ expert +
  off-corridor drift) → 930→1223 needs a categorically better method = **PIMC/search** (engine's own API, native
  4200 rollouts/turn, 99.7% of bank unused; past 1-ply search failed only for weak rollout policy). DECISIONS:
  (1) build PIMC on **Crustle** (our only working pilot = strong rollout prior), NO new decks; (2) do NOT evict v018
  from tracked-2 now (July rank cosmetic; restore champions before 08-11 freeze). PLAN.md updated: new §S phase
  (S1 re-param pimc_agent off dead Iono → Crustle, S2 determinizer w/ meta priors, S3 bank manager, S4 two-arm gate
  [beat own IL prior H2H + Gauntlet-v2 vs reconstructed meta], S5 live probe, S6 → ExIt if gains thin). F2 new-deck
  BC superseded. NEXT = S1. Memory: [[project-search-pivot]], [[project-bc-combo-failure]] (Grimmsnarl added).

- **2026-07-03** — STRATEGY RESET after full recon+audit (details in memory/project_meta). Facts: rank 3199/4102
  (tracked-2 = weak Iono pair 317.7/499.2); meta fully rotated (top-10 = Grimmsnarl ex toolbox ×3 incl #1 tonakaiiii
  1246.7, Cubchoo package ×3, Starmie/Alakazam/Comfey; zero Crustle/Iono/Lucario); onechan1 → rank 206 inactive;
  48 games/day/sub + 10% random since 06-30; organizer released native libs + full engine source. Audit: ExIt loop
  never run; PIMC rollout policy hard-wired to Iono heuristic; v013 regressed vs v012 (38%/200g); v014 never gated;
  v011 missing cg/; champion.txt was stale (v002_lucario). Plan v2 adopted (F0-F5): resubmit v007+v009a (artifacts
  verified+presubmit PASS; first approval attempt blocked — classifier rejected "всё да"; explicit phrase given
  later → **UPLOADED ~10:47 UTC: v009a → ID 54295400, v007 → ID 54295405, both PENDING self-validation**), native
  engine, fresh meta recon → deck decision by data (candidate: Grimmsnarl toolbox), pilot v2 IL, PIMC via
  prior_agent hook, first-ever ExIt run, weekly meta re-recon, freeze 08-11. New gate standard = Gauntlet-v2
  (multi-archetype pool + live probe as final arbiter).

- **2026-06-18 (a)** — v010c_iono IL feature improvement + retrain + gate DONE. VERDICT: FAIL, REGRESSED.
  **Changes made:** (1) Fixed slot37 (`retreat_to_kilowattrel` was always 0 — RETREAT opts have no bench-target fields
  in the option dict). Replaced with `attach_target_is_voltorb` (attaching TO a Voltorb). (2) Added 5 new local
  per-option slots 38-42: `attach_target_is_bellibolt`, `attach_energy_is_lightning`, `attach_to_ready_attacker`,
  `voltaic_chain_would_ko` (Voltorb KO signal), `n_voltorb_on_bench`. FEATURE_DIM: 61 → 66.
  (3) Thin lethal heuristic guard in main.py: if exactly one ATTACK option KOs opp active (slot34 > 0.5), pick it;
  else fall through to IL.
  **feat_diag after (new slots, onechan1 winner MAIN, 11844 option-rows):**
    slot37 attach_target_is_voltorb: 13.7% nonzero (was 0.0% — fixed)
    slot38 attach_target_is_bellibolt: 12.9% nonzero
    slot39 attach_energy_is_lightning: 49.1% nonzero
    slot40 attach_to_ready_attacker: 19.9% nonzero
    slot41 voltaic_chain_would_ko: 4.1% nonzero
    slot42 n_voltorb_on_bench: 65.3% nonzero
  **Offline training (Docker, onechan1 scope, 300 trees, same C6 params):** 
    train 2078 groups (MAIN 915), val 463 groups (MAIN 207). NDCG@1 val=0.713. 
    Top-1 MAIN val: 48.8% (vs option-0 baseline 23.2%) — model looks good offline.
    Feature importance: attach_energy_is_lightning #7 at 5.0%; new slots 37/38/40/41/42 low (<1%).
  **Re-gate v010c vs v009a_crustle:**
    100-game smoke (seed 0): 30W–70L, WR=0.30, Wilson CI [0.219, 0.396] — FAIL
    300-game full (seed 0): 73W–227L, WR=0.2433, Wilson CI [0.198, 0.295] — FAIL (vs v010a's 41% at 100 games)
    Decision breakdown: 10916 model (100%), 4 heuristic, 0 trivial fallback.
  **ONE-LINE VERDICT:** v010c REGRESSED vs v010a (24% vs 41% winrate vs Crustle). The new features increased
  offline metrics but made live play worse — likely model variance on small training set (1122 MAIN decisions)
  plus the new features shifting importance away from what was working in v010a. Iono path is currently
  BELOW the 0.50 decision threshold vs Crustle. **HUMAN DECISION NEEDED:** (A) continue Iono with better data
  collection / architecture changes, (B) fall back to Crustle path (v009a at 52.3% over 300 games), (C) investigate
  why v010a outperformed despite weaker offline features (possibly v010a's model learned something specific
  about Crustle from the random training/val split). Artifacts preserved: v010c_iono/ (new), v010a_iono/ (baseline),
  models/il_iono_c.txt (new model), models/il_iono_a.txt (old model).

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
