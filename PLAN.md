# PLAN — Pokemon TCG AI Battle (execution planner)

> **This file is the single source of truth for execution.** A junior model reads it,
> finds the first unchecked task, does it, then updates the **Status Log** at the bottom.
> Read `CLAUDE.md` first for project facts; this file is the *what to do next*.

Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
Sim deadline: **2026-08-16** → ~2 weeks extra games → final ~2026-09-01.

## Current standing (2026-07-16) — plan v5 (SILVER-FIRST, user-confirmed) — ⚠ SUPERSEDED 2026-07-18 by §J (v039 RECON-DRIVEN); tracked-2 обновлён 07-17 (v038/v037)

- **Цель переформулирована 2026-07-16 (AskUserQuestion): серебро — база, золото — апсайд.** ⚠ ПОРОГИ ИСПРАВЛЕНЫ
  по полному лидерборду (5118 команд, 07-16, recon sonnet-агентом; прежние цифры в этом абзаце были неточны и путали
  bronze/silver): **gold rank11 μ1132.4** (top-20 rank20 μ1087.4), **silver rank103 μ993.2**, **bronze rank256 μ923.4**,
  медиана (rank2560) μ658.4; мы rank2605 μ654.3. Наш исторический пик **v007_crustle (submission 53783594) = live 930.7
  = УРОВЕНЬ БРОНЗЫ, НЕ СЕРЕБРА** (до серебра 993.2 не хватает +62.5μ; прежняя «+5.4 выше серебра» — ошибка метки).
  Рамка цели — реалистичный floor = бронза (вернуть пик ~930 свежим топ-10 Crustle-пилотом), stretch = серебро (клон
  топ-10 пилота μ1100+ с хорошим BC-capture), far upside = золото (План D RL). Полный план — **§I**.
- **Мы сейчас (обновлено 2026-07-17):** tracked-2 = {`v038_plan_d_ppo_poolfix` (submission 54791599, залит 07-17,
  PENDING — diverse self-play pool, WR vs alakazam 0.41→0.66 на честном гейте), `v037_plan_d_ppo_hidden`
  (submission 54789080, залит 07-17, μ674.2, hidden-obs PPO)}. Предыдущие v030/v007_resub/v033_resub/v034/v035/v036
  вытеснены из tracked-2 серией сабмитов 07-16..07-17. Наш исторический пик v007_crustle = live 930.7 = бронза.
- **Почему пик не держится:** мета ротировалась анти-Crustle (§F1 recon 07-03: Grimmsnarl/Munkidori/Cubchoo/Starmie/
  Alakazam доминируют) — `field_viability.py` (07-12j) показал Crustle проваливает Alakazam (0.55) и Cubchoo (0.60):
  Alakazam Powerful Hand кладёт damage counters мимо prevent-shield Mysterious Rock Inn (независимо подтверждено на
  форуме, тред 726485 «[Solved]»). Deck-tech фикс уже построен и гейтился (`v023_crustle_techD`, submission 54590285:
  Alakazam 0.35→0.52, Cubchoo 0.63→0.82, overall паритет 0.9106 vs 0.9118) — **live-вердикт не получен**, пробу
  оборвали на 53 играх ради переключения на v024 NN-эксперимент. ⚠ ОБНОВЛЕНО 2026-07-18: B3a (Alakazam-патч) закрыт
  4-й раз подряд (v012/v017/v036×3/v033) — v023-techD «открытый хвост» более не актуален. Новый угол B3a = hand-disruption
  (Unfair Stamp 1080 + Xerosic 1197, +23.9пп локально vs Alakazam из recon 2026-07-18) — см. §J1, НЕ §I Путь 0.
- **★★ Новый актив 2026-07-16: официальный ежедневный датасет топ-эпизодов организатора**
  (`kaggle/pokemon-tcg-ai-battle-episodes-{date}`, форум 709160, отбор по среднему рейтингу участников, медиана
  μ1125.9 на 07-15) — снимает 429-блокер плана клонирования (`project_clone_majkel_next.md` в памяти). Форум
  717697 «RL journey»: независимый участник дошёл до серебра чистым self-play PPO за ~1 день на одной GPU +
  рекомендует BC-претрейн политики на сильных демонстрациях перед RL (1/100 компьюта) — усиливает План D (§I Путь 2).
- **Все прежние закрытые пути (§H3-4-re KILL PIMC, BC-потолок 930-1000, deck-tech соло-паритет и т.д.) остаются
  закрытыми** — §H архивируется ниже, детали не переоткрываются без нового аргумента (см. таблицу «Отвергнутые пути»).

---

## Archive — Current standing (2026-07-11) — plan v4 (HYBRID PIMC, superseded by v5)

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

**NEXT ACTION (2026-07-18, plan v039, см. §J):** Текущий актуальный план = **§J (RECON-DRIVEN)** на основе 6 Kaggle-ноутбуков
2026-07-18. §I (plan v5) и §H (PIMC) архивированы. **Фокус:** ждать live-вердикт `v038_plan_d_ppo_poolfix` (submission 54791599,
PENDING, ≥150-250 игр) → §J0 интерпретация по бендам (myso: Alakazam 38-45% топа, Marnie Grimmsnarl топ-2, Archaludon 0% в 1100+)
→ §J5 выбор ветки: **J1** hand-disruption vs Alakazam (Unfair Stamp 1080 + Xerosic 1197, +23.9пп локально) / **J2** PPO-улучшения
поверх v038 / **J3** replay-mining 4 архетип-кандидатов. Plan D (PPO на abiolatti) дал 3 залитых сабмита (v033_resub/v037/v038);
B2 god-mode гипотеза ОПРОВЕРГНУТА 07-17.
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

## H — PLAN v4: HYBRID PIMC (CLOSED 2026-07-16: §H3-4-re KILL PIMC honest on 700 games → superseded by §I)

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
| Клон новой топ-деки как обход потолка BC | v018 453–484 при эксперте kazuki 1083 (~40% силы); PokéForge-клон 802 = их потолок; **v032b Crustle+MegaKangaskhan клон (2026-07-16): H2H 0.40 vs v009a при val_top1 0.487 — Kangaskhan-aware фичи не пробили BC-потолок на setup-деке** |
| Наращивание ОБЪЁМА IL-данных | v008: 3× данных → mirror 0.394, регресс |
| Каунтер-гейт vs свой чемпион как критерий сабмита | v012: 0.70 локально → 499.2 live |
| Ручные патчи поверх сильной политики без поиска | PokéForge: 0 измеримого прироста; наши v012→v013 (38%) |
| 1-ply / rollout-search ≈ своя же политика | v003: три гейта ≈0.50 (06-17) |
| PIMC с rollout ОБЕИХ сторон = свой prior | S4b: −0.40 vs Alakazam, −0.15 vs Grimmsnarl (чинится H3-1; как было — не повторять) |
| Полный ExIt/AlphaZero-цикл с нуля | PokéForge self-play/DAgger/A2C провал; ~36 дней до дедлайна; только ExIt-lite поверх рабочего PIMC |
| IS-MCTS как первая реализация | детерминизацию всё равно требует API движка; merge деревьев между процессами дорог (лит-рекон 07-11) |
| Ре-сабмит как «сохранение» рейтинга | рейтинг не наследуется (форум 714189); v007-resub карабкался 600→733 заново |
| Микротюнинг деклиста под замороженный пилот | Phase-3 гаунтлет: все recount-варианты статистически неотличимы от базы |

## I — PLAN v5: SILVER-FIRST (ARCHIVED 2026-07-18 — superseded by §J v039 RECON-DRIVEN; был PRIMARY path 2026-07-16, user-confirmed)

> Цель: **серебро (μ993.2, rank103/5118) — база; золото (μ1132.4, rank11) — апсайд**, если path 2 выстрелит.
> ⚠ ПОРОГИ ИСПРАВЛЕНЫ (полный лидерборд 07-16): μ923.4 = **BRONZE** (rank256), НЕ серебро; до серебра +62.5μ.
> Якорь: наш исторический пик v007_crustle = live 930.7 (submission 53783594) = **уровень БРОНЗЫ, НЕ серебро**.
> Полное обоснование, цитаты форума/памяти — см. чат-сессию 2026-07-16 (эта секция — её итог).

### I0 — Гигиена (низкая стоимость, вести параллельно со всем остальным)

- [~] **I0-1** Досмотреть `v030_time_budget` (submission 54752121) до 150-250 игр (было 47 на 07-16 ~13:40 UTC,
      темп ~125 игр/день) → keep/rollback по числу игр, не по часам ([[project-rating-noise]]).
      **ОТМЕНЕНО 2026-07-17 — v030 вытеснен из tracked-2 серией сабмитов v033_resub→v034→v035→v036→v037→v038.**
- [~] **I0-2** Довершить live-вердикт `v023_crustle_techD` (submission 54590285) — гейтился Alakazam +0.167,
      Cubchoo +0.183, overall паритет 0.9106/0.9118; live-проба оборвана на 53 играх. Нужен новый сабмит
      (слот + «ок» пользователя) ИЛИ дождаться естественного окна tracked-2.
      **ОТМЕНЕНО 2026-07-18 — B3a закрыт 4-й раз подряд, v023-techD неактуален; новый угол B3a = §J1 hand-disruption.**
- [ ] **I0-3** Sanity-check: подтвердить, что наш пайплайн (IL/MCTS) действительно предпочитает первый ход при
      выборе first/second (форум 723591: first WR 55.2% vs second 44.8%, +10 очков). Нулевая стоимость, не
      гарантированный выигрыш — просто проверка, что не отдаём бесплатные +10 очков.

### I1 — Клон через официальный датасет топ-эпизодов (дни, ожидаемый потолок ~900-1000)

- [ ] **I1-1** Добавить `kaggle/pokemon-tcg-ai-battle-episodes-{07-14..07-16}` как input-датасет в Kaggle-ноутбук
      (НЕ тянуть 21GB/день локально). Манифест дней — `kaggle/pokemon-tcg-ai-battle-episodes-index`.
- [ ] **I1-2** Найти игры текущего живого топа (Majkel1337 μ1285.5, Oshbocker, Luca, THIRD PTCG Club, Taichicchi,
      懒惰的金枪鱼, Rmy, kashiwashira, junlee789, bono — свежие имена из лидерборда 07-16, НЕ протухшие архетипы
      recon 07-03/07-15) grep'ом по именам агентов внутри JSON — без единого точечного REST-запроса, без 429.
- [x] **I1-3** Выбрать ОДИН ещё живой архетип, вытащить deck+пилот, строить IL-клон на тысячах партий (не 300).
      **Строго фильтровать по архетипу** — не повторять ошибку v008 (3× данных без фильтра = регресс,
      [[project-il-data-quality]]). Помнить [[project-bc-combo-failure]]: комбо/сетап-деки (Alakazam-класс)
      проваливают BC хуже, чем midrange aggro.
      ✅ **CLOSED 2026-07-16 (s) — КЛОН-ПУТЬ ЗАКРЫТ, BC-потолок подтверждён 5-й раз.** Архетип = Crustle+MegaKangaskhan ex
      (топ-10 пилот 懒惰的金枪鱼/__Taichicchi__, 726 эпизодов из официального датасета 07-14/07-15, deck-signature {345,756}+winner).
      3 итерации фичей: v1 (features_v009a Crustle-спец) val_top1 0.436 / H2H 0.24; v2 (больше capacity) 0.447; **v3
      (features_clone Kangaskhan-aware, FEATURE_DIM=75: is_kangaskhan, retreat_to_kangaskhan, is_hand_trimmer, is_heal_card,
      is_heros_cape, is_stadium, is_boss_orders, has_kangaskhan_on_bench, my_active_is_kangaskhan, …) val_top1 0.487
      (стена 0.45 пробита, +4пп), best_iter=83, train≈val (переобучения НЕТ — потолок = сигнал фичей).** Гейт anchored_pool
      (600 игр, 0 exc): **H2H vs v009a WR 0.40 [0.348,0.458] — ЗНАЧИМО хуже чемпиона** (улучшение 0.24→0.40, но не parity);
      vs v001_random 0.80 (RED-FLAG passed). Importances: type_ABILITY 18%, g_turn_norm 14%, **card_is_switch 4.6% (#4 —
      единственная НОВАЯ дек-фича в топ-15)**; Kangaskhan/hand-trimmer/heal/cape/stadium/boss identity **не вошли в топ-15**
      — слишком разрежены в 726 эпизодах, ранжер учится на общих сигналах (ABILITY/turn/energy), а не на distintive
      trainer-decisions. Это ровно BC-потолок на setup-деке (2 эволюции + Mega + disruption): [[project-bc-combo-failure]].
      **Вердикт: даже parity = бронза (930.7), не серебро (+62μ). Клон-путь закрыт; v032b НЕ сабмитится.** Фичи/модель
      (features_clone.py, il_crustle_clone_v3.txt) оставлены как потенциальный BC-warmstart для Плана D (§I2-1). Артефакты:
      `src/il/features_clone.py`, `models/il_crustle_clone_v3.txt`, `submissions/v032b_crustle_clone/`,
      `results/gates/v032b_gate_state.json`. Урок в памяти [[project-clone-kangaskhan-bc-ceiling]].
- [ ] **I1-4** Гейт vs текущий baseline (Docker-образ уже с `torch`, см. docker/Dockerfile) → live-probe при «ок».
      *(I1-3 закрыт без кандидата на live — I1-4 неактуален для этого архетипа; переоткроется если появится новый BC-friendly архетип.)*

### I2 — План D: PPO на abiolatti + BC-warmstart (уже идёт, средний риск, единственный путь к золоту)

Статус на 07-18: D0 (валидация engine vs cabt) PASS, D1 (порт PPO-инфры) COMPLETED, **D2 (self-play GPU) COMPLETED** —
3 PPO-сабмита залито: `v033_plan_d_ppo` (54779509, 07-17, ре-сабмит 5M-чекпойнта), `v037_plan_d_ppo_hidden` (54789080,
07-17, hidden-obs, μ674.2), `v038_plan_d_ppo_poolfix` (54791599, 07-17, diverse self-play pool, WR vs alakazam 0.41→0.66
на честном гейте, PENDING). B2 god-mode гипотеза ОПРОВЕРГНУТА 07-17 (hidden-obs WR 0.645 ХУЖЕ старого god-mode 0.792 на
честном гейте; RUN B рецепт закрыл разрыв на внутреннем движке, но НЕ на решающем cabt). tracked-2 = {v038 PENDING, v037 μ674.2}.
Подробности — [[project-plan-d-medal]], [[project-v038-poolfix-gate-success]], [[project-b2-godmode-hypothesis-falsified]].

- [ ] **I2-1** Встроить BC-претрейн политики на официальном датасете эпизодов (§I1) перед/во время PPO —
      прямая рекомендация форума (тред 717697: «1/100 компьюта на валидацию + резко ускоряет RL»). Снижает риск
      повторить недообучение v1/v2 на малом числе шагов.
- [ ] **I2-2** Продолжать D2 self-play, чекпойнты каждые 1M шагов, следить за WR vs random по мере роста шагов.
- [ ] **I2-3** D3 — инференс через `torch` в `main.py` (live-safe, MCTS на cabt в инференсе, НЕ custom engine) —
      после того как D2 покажет уверенный прогресс.

**Внешнее подтверждение важности этого пути:** форум-тред 717697 — независимый участник, чистый self-play PPO без
знания игры, дошёл до серебра за «4 часа тренировки → топ-30%, ещё тренировки → серебро» на одной GPU (~45 игр/сек).
Это не только путь к золоту при большом риске — это самостоятельная ставка на серебро с уже частично оплаченной
инфраструктурой (GPU арендован, engine провалидирован).

### I3 — Скейлинг v030 MCTS (низкий приоритет, дешёвое продолжение) — ⚠ ARCHIVED 2026-07-17: v030 вытеснен из tracked-2

- [ ] **I3-1** Search помогает слабым базам (+11pp, [[project-search-vs-rl-ceiling]]) — база v030 (ExIt+GBDT-value)
      относительно слабая → есть небольшой потенциал докрутки search_count/opponent-model, но не в фокусе.

### I4 — Сравнение и заморозка

- [ ] **I4-1** 08-01..08-08: сравнить кандидатов (I0/I1/I2/I3) по живому μ (150-250 игр каждый), выбрать 2 лучших.
- [ ] **I4-2** 08-09..08-11: зафиксировать tracked-2, никаких новых экспериментов (гвардрейл freeze, ниже).
- [ ] **I4-3 Risk floor:** если к ~08-05 ничего не обгоняет baseline — в tracked-2 остаётся лучший из уже
      живо-проверенных (v007/v009a/v023techD) — стабильность важнее непроверенной попытки перед заморозкой.

---

## J — ПЛАН v039: RECON-DRIVEN (после v038 live-вердикта; источники — 6 Kaggle-ноутбуков 2026-07-18)

> **Контекст:** v038 (PPO diverse self-play pool, submission 54791599) уже залит и PENDING — править нельзя,
> только перезаливать новым сабмитом (съест слот + вытеснит v037 из tracked-2). Поэтому все находки ниже —
> материал для **следующей** версии (v039), строится ПОСЛЕ live-вердикта v038 (≥150-250 игр). Источники:
> `myso1987/leaderboard-deck-meta-by-score-band`, `ygkojima/ptcg-strategy-report-v2`, `naoto714` ×2 (Kangaskhan),
> `premaananda108/daily-meta-deck-win-rate-analysis`, `beicicc/ptcg-public-experiment-snapshot-jul17`.
> Сводка находок в памяти: [[reference-kaggle-notebooks-recon-2026-07-18]].

### J0 — Интерпретация v038 live-вердикта (gating, бесплатно)
- [ ] **J0-1** Когда v038 наберёт ≥150 игр — смотреть WR **по рейтинговым бендам**, не общий (ygkojima:
      Alakazam vs Kangaskhan = 61% в band ≥1100, но 28.7% в 800-1099 — смешивание бендов скрывает разворот).
      Если v038 тонет именно в верхнем диапазоне — это состав оппонентов, не баг агента.
- [ ] **J0-2** Не паниковать на малом n: σ_aa = 41.9pt на идентичных сабмитах (ygkojima, наше
      [[project-rating-noise]]). μ670-720 на 50 играх = шум, ждать 150-250.

### J1 — B3a-фикс Alakazam: hand-disruption (новый угол, СРЕДНИЙ приоритет)
> Alakazam = монополист топа: 24% (600-699) → 38% (900-999) → 45% (1000-1099) → 43% (1100+) (myso1987).
> В нашем диапазоне — каждый 3-й оппонент. v038 уже дал WR vs Alakazam 0.41→0.66 на честном гейте —
> но B3a-дыра (Alakazam закрыт 4 раза подряд без патча: v012/v017/v036×3/v033) может стоить верхнего диапазона.
- [ ] **J1-1** Override/deck-tech: **Unfair Stamp (1080) + Xerosic's Machinations (1197)** — hand-disruption
      вместо танк-Hero's Cape. Логика: `Powerful Hand` = +2 dmg/карта в руке оппонента → отбирать руку,
      не наращивать HP. У naoto714 локально +23.9пп vs Alakazam (23.4%→47.3%). ⚠ ПРЕДОСТЕРЕЖЕНИЕ: его локальный
      гейт vs одного Alakazam-прокси может переоценивать; наш retreat-патч 3/3 A/B не дал улучшения на
      v017_alakazam_heur ([[project-v036-forensic-findings]]). → проверять ТОЛЬКО с live-форензиком 1-го матча,
      не доверять локальному гейту vs Alakazam.
- [ ] **J1-2** Альтернатива: Koraidon Shred (1 копия, игнорирует эффекты на active) — у naoto714 1-7 live,
      слабо. Низкий приоритет.

### J2 — PPO-улучшения для следующего retrain (ВЫСОКИЙ, дёшево)
- [ ] **J2-1** **Нелинейная prize-diff→WR кривая** в value-head / reward shaping (ygkojima, N=70,303):
      плато после +3 (80.5%→81.8%), асимметрия ниже 0. Заменить линейный prize-diff на измеренную кривую
      в leaf eval PIMC и в reward PPO. Прямо касается нашего потолка.
- [ ] **J2-2** **Marnie Grimmsnarl** — топ-2 архетип (14-15% в 1100+, растёт с рейтингом), НЕ был на радаре
      (myso1987). Проверить, входит ли Marnie в diverse-pool v038; если нет — добавить в пул v039.
- [ ] **J2-3** **Mega Abomasnow ex** — 774 игры, WR 43.4%, крупный кластер поля (premaananda108). Добавить
      в archetyp-пул retrain и в forensic-чек.
- [ ] **J2-4** **BC-баги D01/D03** (ygkojima): D01 turn-index rescaling (train 0-N, inference 5× scale →
      early-game misrecognition); D03 own-bench composition absent → placement не выучивается. Проверить
      в наших BC/IL пайплайнах (`src/il/features_*.py`).

### J3 — Replay-mining: 4 новых архетип-кандидата (СРЕДНИЙ, при следующем цикле разведки)
- [ ] **J3-1** Tien Search-Alakazam hardened (beicicc s132, 955.6, "prior 1034.6") — search+learned value,
      отличается от нашего tientrum reverse-engineer. Кандидат №1 на реплей-майн.
- [ ] **J3-2** LLCC Crustle AttackFix (s107, 964.3) — Crustle-вариант выше нашего пика 930.7; "AttackFix"
      может быть фикс нашего retreat/attack бага.
- [ ] **J3-3** Pilkwang Archaludon metal-tempo (s052, 1030.6) и Tomato Archaludon anti-Starmie (s045, 964.1,
      стабильный 855-948) — два archaludon-листа отличных от нашего v036 (831.9 live).
- [ ] **J3-4** ⚠ Все кандидаты в зоне 955-1030 с большой дисперсией (rating noise, beicicc: идентичный пакет
      → 473/1027/822/591/835/662/473). Не спешить — наш пик уже в бронзе.

### J4 — Инструментарий (низкая стоимость, reuse)
- [ ] **J4-1** `deck hash = md5(sorted(deck_ids))[:8]` (premaananda108) — компактный ключ архетипа для
      агрегации эпизодов между днями; добавить в `scripts/` deck-scan.
- [ ] **J4-2** Lethal solver как отдельный модуль (ygkojima EXP-030: 22.8% attack-decisions recoverable
      правильным lethal-solver'ом). У нас PIMC/ExIt, но явный арифметический lethal-предиктор может быть
      дешёвым усилением. Низкий приоритет.

### J5 — Решение по v039 (после J0)
- [ ] **J5-1** Если v038 live удержит WR vs Alakazam ≥0.55 и общий μ ≥800 → v039 = J2 (PPO-улучшения:
      нелинейная prize-кривая + Marnie/Abomasnow в пул) как эволюция v038.
- [ ] **J5-2** Если v038 live провалится именно vs Alakazam → v039 = J1 (hand-disruption deck-swap версия)
      как отдельный сабмит с live-форензиком.
- [ ] **J5-3** Если v038 live ≈ паритет/шум → J3 (реплей-майн Tien Search-Alakazam / LLCC Crustle AttackFix)
      как источник нового BC-warmstart для Плана D.

### J-fix — ПРИНЯТЫЕ ПРАВКИ v039 (2026-07-18, user-approved; БЕЗ сабмита, только файлы)

> Решение юзера 2026-07-18: для self-колоды принять рекомендацию #1 (Crustle+MegaKangaskhan эталон);
> для PPO pool принять правки #4,#5,#6,#7. Подробно + 60-карточные дек-листы —
> [[project-v039-deck-and-pool-recommendations]]. Контекст: наша self-дека v038 сломана (8 покемонов /
> 31 энергия / 21 trainer; live-симптомы: mulligan'ы, deck-out от навешивания по 10 энергий); все топ-250
> имеют 7-15 энергий. Живой топ-Crustle = 13P/13E/34T. PPO pool v038 НЕ содержал живого топ-Crustle вообще.

**Сделанные файловые правки (артефакты готовы, PPO retrain НЕ запущен — ждёт GPU + «ок»):**

- [x] **J-fix-1 (self-дека, рекомендация #1)** — `submissions/v039_crustle_megakanga/` создана как копия
      v038 (main.py + cabt_conv.py + nn_plan_d.py + cg/ + ppo_d2runDfull_final_s5013504.pth), `deck.csv`
      ЗАМЕНЕН на Crustle+MegaKangaskhan эталон (rank 3, 7 команд топ-250): 13P/13E/34T, ядро
      Dwebble×4+Crustle×4+**Mega Kangaskhan ex(756)×4**+Shaymin(343)×1, энергии 1 базовая {G}+12 специальных
      (Mist 11×4 / Spiky 14×4 / Grow Grass 18×4). main.py НЕ тронут (тот же чекпойнт → это deck-swap
      эксперимент: пилот, обученный на старой сломанной деке, играет новую деку без retrain).
- [x] **J-fix-2 (deck-файлы эталонов)** — `decks/v039_pool/` создана, 4 файла по 60 карт:
      `crustle_megakangaskhan.csv` (=self, 13P/13E/34T), `alakazam_A.csv` (19P/7E/34T, вариант A 18 команд),
      `grimmsnarl_live.csv` (18P/10E/32T, живой топ-эталон), `mewtwo.csv` (15P/13E/32T, Team Rocket's Mewtwo ex).
- [x] **J-fix-3 (POOL_REGISTRY, правки #4-7)** — `scratchpad/gold/plan_d/d1_train/train_ppo_v2.py`
      POOL_REGISTRY обновлён (старый блок сохранён как комментарий `# POOL_REGISTRY v038 (PRE-v039)`):
      - self → `decks/v039_pool/crustle_megakangaskhan.csv`, вес **0.30** (НОВАЯ self-дека, 31-энергия убрана)
      - alakazam → `decks/v039_pool/alakazam_A.csv`, вес **0.30** (было 0.20; живой эталон A вместо v017)
      - grimmsnarl → `decks/v039_pool/grimmsnarl_live.csv`, вес **0.12** (было 0.125; живой эталон вместо v018)
      - **crustle_mirror → `decks/v039_pool/crustle_megakangaskhan.csv`, вес 0.12** (НОВЫЙ opponent — жилого
        топ-Crustle в pool не было; 11% топа)
      - **mewtwo → `decks/v039_pool/mewtwo.csv`, вес 0.06** (НОВЫЙ opponent)
      - archaludon → `submissions/v036_archaludon_patched/deck.csv`, вес **0.05** (было 0.125; 3% топа, 0% в 1100+)
      - lucario → `submissions/v035_makthanithin/deck.csv`, вес **0.05** (было 0.125; исчезает в 1000+)
      - **impidimp УБРАН** (дублирует grimmsnarl-семью)
      - Сумма = 0.30(self)+0.30+0.12+0.12+0.06+0.05+0.05 = 1.00.
      - ⚠ **ФОРМАТ**: `POOL_REGISTRY` — это `dict[str,str]` (имя→путь к deck.csv), **веса в нём НЕ хранятся**.
        Веса передаются через CLI-флаг `--opp-decks` при запуске обучения. Субагент обновил пути в dict
        (чтобы `crustle_mirror`/`mewtwo` стали резолвимыми метками) + пример в help-тексте/docstring на
        v039-веса. Сам по себе обновлённый POOL_REGISTRY pool НЕ активирует — только делает метки валидными.

**NEXT для другого агента (по порядку):**

- [x] **J-fix-N1 (бесплатно, без GPU)** Локальный гейт deck-swap: `submissions/v039_crustle_megakanga/`
      (новая дека + старый чекпойнт v038) vs {v007,v009a,v033,v036}, N=300, honest gate (cg_native, non-god-mode,
      seed=20260716). Скрипт `scripts/gate_expanded.py`. **ВЕРДИКТ 2026-07-18: deck-swap БЕЗ retrain = КОЛЛАПС.**
      WR v039 vs v009a = **0.153** [0.117,0.199] (v038 baseline 0.700 → Δ −55пп); vs v007 0.160 (v038 0.660 → Δ−50пп);
      vs v033 0.093 (v038 0.430 → Δ−34пп); vs v036 0.200 (v038 0.200 → Δ~0). 0 exceptions, nn_decisions ~22K/300игр
      = ~73 хода/игру → пилот ИГРАЕТ длинные партии, но проигрывает (не mulligan-автопроигрыш). Дека структурно
      валидна (Dwebble×4 базовые для старта) — коллапс = deck-policy entanglement: NN пилота v038 обучена на
      card-ID старой деки (1212/1235/31×энергия), при свопе на 756/1087/1122/1197/1225/1227 её выходы неадекватны.
      WR vs v009a 0.153 ≪ порога 0.55 → **deck-swap без retrain НЕ ценен, fine-tune (J-fix-N2) ОБЯЗАТЕЛЕН.**
      Файлы: results/gates/{v038_vs_v007_crustle,v039_vs_v007_crustle,v039_vs_v009a_crustle,v039_vs_v033_plan_d_ppo,
      v039_vs_v036_archaludon_patched}.json. Память: [[project-v039-deckswap-no-retrain-collapse]].
- [x] **J-fix-N2 (GPU, одобрено юзером 2026-07-18)** PPO fine-tune с чекпойнта v038 на НОВОМ pool
      (J-fix-3) + НОВОЙ self-деке (J-fix-1). 3M шагов (увеличено с 1-2M — юзер разрешил, радикальный deck-swap
      требует больше шагов; kl-anchor-decay 3M). vast.ai RTX 5090, ~110s, ~29000 steps/s. Чекпойнт
      `ppo_v039ft_final_s3014656.pth`. Pool_wr в обучении росли (alakazam 0.26→0.78, entropy 0.117→0.187,
      mean_ret +0.59→+0.376, p0_term_wr 0.45→0.73) — НО локальный гейт провален (см. J-fix-N3).
- [x] **J-fix-N3** Локальный гейт v039_ft (new-ckpt+new-deck) vs 4 оппонентам + факторный 2×2.
      **ВЕРДИКТ 2026-07-18: fine-tune = CATASTROPHIC FORGETTING.** Гейт vs v009a
      = 0.423 [0.369,0.480] (v038 0.700 → Δ−27.7пп); vs v007 0.428 (Δ−23.2пп); vs v033 0.230 (Δ−20.0пп);
      vs v036 0.350 (Δ+15.0пп единственный плюс). 0 exceptions. Факторный 2×2 (WR vs v009a): old-ckpt+old-deck
      = 0.700 (v038 baseline); old-ckpt+new-deck = 0.153 (no-retrain); new-ckpt+new-deck = 0.423 (ft);
      **new-ckpt+old-deck = 0.418** (ЭКСПЕРИМЕНТ, submissions/v039_ft_olddeck/). new-ckpt+old ≈ new-ckpt+new
      (CI перекрываются) → возврат старой декы НЕ восстановил v038 → **виновато ОБУЧЕНИЕ/POOL, не дека**
      (catastrophic forgetting). Глубинная причина: pool v039 рассогласован с гейт-полем (pool-alakazam
      alakazam_A+mirror/random ≠ v009a Crustle IL ≠ v033 Alakazam PPO); pool_wr 0.78 в обучении не переносится
      на гейт 0.42. Политика переучилась на лёгких pool-оппонентах (mirror 42% + mewtwo/lucario/archaludon
      0.76-1.0). Файлы: results/gates/v039ft_vs_*.json + v039ft_olddeck_vs_*.json, results/gates/v039_finetune.log.
      Память: [[project-v039-finetune-forgetting-2x2]].
      **САБМИТ live-probe 2026-07-18 (юзер одобрил):** v039_ft → submission **54793514** PENDING. Live-safety
      smoke PASS. tracked-2 стало {v039_ft PENDING, v038 μ685.1}; v037 вытеснен. ★ v038 live μ обновился до
      685.1 (середняк, ниже пика 930.7) — подтверждает generalization gap (локальный 0.70 не→live-прорыв),
      что и обосновывает live-probe v039_ft несмотря на локальный регресс. Ждать 150-250 live-партий.
      Память: [[project-v039ft-shipped-v038-live-685]].
- [ ] **J-fix-N4 (опционально, после v039 live)** Если deck-swap (J-fix-N1) дал плюс, а fine-tune (J-fix-N2)
      дал дополнительный плюс — сложить: v040 = fine-tuned пилот + обновлённый pool + hand-disruption
      override против Alakazam (§J1-1, Unfair Stamp 1080 + Xerosic 1197) как отдельный эксперимент.
- [ ] **J-fix-N5 (v042_energy_fix, ждёт GPU)** После провала v039_ft/v040_fresh/v041_extpool (все хуже v038
      0.700 vs v009a: 0.423/0.348/0.303) — юзер одобрил комбинированный фикс вместо очередного fresh-start:
      (a) НОВЫЙ Phi-терм `--shaping-w-energy` в `train_ppo_v2.py::extract_mover_potential` — штрафует потерю
      энергии на активном покемоне (retreat/KO-замена на E0), напрямую бьёт failure modes 1+2 без спец-детекции
      CG_OPTION_RETREAT/TO_ACTIVE; (b) НАЙДЕН баг KL-anchor: v039_ft анкорился к СТАРОМУ BC-warmstart (не к
      v038!), т.к. `--warmstart-policy` указывал на `warmstart_policy_hidden_full.pth`, а v038 грузился через
      `--resume-ckpt` — ref_model снимается ДО resume-ckpt (код train_ppo_v2.py:~830-850, комментарий "so the
      anchor stays to the ORIGINAL warmstart"). Это правдоподобно объясняет catastrophic forgetting лучше, чем
      прежняя гипотеза "pool рассогласован с гейт-полем". Фикс: `--warmstart-policy` теперь указывает НАПРЯМУЮ
      на v038 чекпойнт (`submissions/v038_plan_d_ppo_poolfix/ppo_d2runDfull_final_s5013504.pth`), БЕЗ
      `--resume-ckpt` — тогда ref_model = v038 корректно; (c) opp-pool веса сдвинуты в сторону реальных
      оппонентов: v009a 0.22, v007 0.10, v033 0.18, v036 0.10, v034 0.05, v035 0.05 (сумма 0.70, было 0.27),
      self 0.20, random 0.10 (было 0.73) — дека НЕ трогается (тот же `decks/v039_pool/crustle_megakangaskhan.csv`,
      соответствует подтверждённому top-250 профилю P≈13/E≈13); (d) шаги подняты 5M→10M (юзер разрешил больше
      шагов, если v041 недоучился — но диагноз v041 показал ПЛОСКИЙ pool_wr весь прогон, не восходящий тренд,
      так что 10M — это буфер на случай другой динамики с новым pool/shaping, не гарантия). KL-anchor coef
      поднят 0.005→0.02 (decay до floor 0.003 за 6M шагов), exclude-types="12" (RETREAT) сохранён, чтобы новый
      energy-Phi term мог менять retreat-поведение не борясь с анкером. Скрипт готов:
      `scratchpad/gold/plan_d/d1_train/server_train_v042_energy_fix.sh`. Требует свежий GPU-бокс (vast.ai,
      старый мёртв) с синком submissions/{v009a_crustle,v007_crustle,v033_plan_d_ppo,v034_romanrozen_clone,
      v035_makthanithin,v036_archaludon_patched,v038_plan_d_ppo_poolfix}/, decks/v039_pool/,
      warmstart_policy_hidden_full.pth больше НЕ нужен (заменён на v038 ckpt). Ждёт GPU + запуска юзером.
      **РЕЗУЛЬТАТ 2026-07-18/19 (свежий GPU-бокс vast.ai, инфра пересобрана с нуля — ptcg_engine из исходников
      с локальными патчами, cg/-движок и данные/веса дозалиты, pandas/lightgbm доустановлены):** 10.03M шагов,
      ~7000-8000 шагов/с, 0 exc/ill у всех 6 внешних оппонентов весь прогон. Локальный гейт (eval_vs_pool.py,
      N=150 детерминированных, та же методология что v038-v041): **v007 0.533, v009a 0.503 (цель ≥0.50
      ДОСТИГНУТА), v033 0.267, v036 0.367** — лучший результат из ВСЕХ попыток на новой деке (v039_ft/v040/v041
      все были 0.30-0.43 на v009a), v007/v009a вернулись к уровню старого v038-бейзлайна (0.660/0.700) НА
      НОВОЙ деке (снимающей deck-out риск). v033 (Alakazam-style PPO) остаётся структурной дырой (B3a),
      energy-shaping её не чинит — другой паттерн. Training-time pool_wr вышел на плато к iter~240/306, гонять
      дальше сейчас смысла не было (не тренд, шум) — юзер разрешил продолжать ЕСЛИ тренд растущий, но не стали,
      т.к. плато. Чекпойнт: `scratchpad/gold/plan_d/d1_train/ckpts_v042_energy_fix/ppo_v042ef_final_s10027008.pth`,
      гейт `results/gates/v042_energy_fix_vs_pool.json`. НЕ засабмичено (ждёт решения юзера — сабмитить/фикшить
      v033 отдельно/что-то ещё).
- [x] **J-fix-N6 (v043_alakazam_mirror_fix) ЗАЛИТ 2026-07-18.** Уточнение: "v033" в гейте — это НЕ Alakazam,
      а наш собственный PPO-Crustle-зеркальный чекпойнт (терминологическая путаница в предыдущем отчёте
      исправлена). Настоящий Alakazam (карта 743, Powerful Hand OHKO) добавлен в pool через
      `v017_alakazam_heur` (наш эвристический клон) вес 0.16; v033 вес поднят 0.18→0.28. Дека: 1x Hero's Cape
      → 1x Unfair Stamp (1080, hand-disruption после KO). **Найден баг:** первая версия с 2x Unfair Stamp
      тренировалась 10M шагов "чисто" на ptcg_engine (0 exc), но карта ACE-SPEC max-1-копия — честный
      cg_native-гейт мгновенно топил все игры (draw, 0 ходов) БЕЗ учёта в exc-счётчиках; обучение полностью
      перезапущено с легальной 1x-версией. Память: [[project-ptcg-engine-deck-legality-gap]]. Результат N=150:
      v017(Alakazam)=**0.673 фикс**, v009a=0.490, v007=0.433, v033=**0.240 не сдвинулся** (был 0.267 у v042
      несмотря на удвоенный вес пула — вероятно не pool-weight-рычаг), v036=0.323. v007/v036 просели vs v042
      (funding cost). Смешанный локальный результат, юзер решил залить: submission **54808614 PENDING**.
      tracked-2 стало {v043_alakazam_mirror_fix PENDING, v041_extpool μ514.5} — v039_ft (μ625.5) вытеснен.
      Память: [[project-v042-energy-fix-gate-success]].

- [x] **J-fix-N7 (v044_archaludon_abomasnow) ЗАЛИТ 2026-07-18.** Submission **54808955 PENDING**. tracked-2
      стало {v044_archaludon_abomasnow PENDING, v043_alakazam_mirror_fix 637.1} — v041_extpool (514.5)
      вытеснен. Юзер дал 1 час полной свободы изучить
      v034/v036 (лучшие живые результаты 865.7/827.3) и предложить улучшение. Находка: v034/v036 —
      детерминированная matchup-aware эвристика (НЕ RL), уже имеет sophisticated ПРЕДИКТИВНУЮ Alakazam-модель
      (`_estimate_alakazam`), принципиально лучше нашего reactive-retreat подхода. Попробовал Hero's Cape→
      Unfair Stamp своп (перенос идеи из v043) — ЛОКАЛЬНО похоже на регресс на v009a/v007/v033, но найден
      шумовой пол гейта ~±8пп на N=150 (5 повторных прогонов ОДНОГО чекпойнта дали разброс 0.760-0.840) —
      отклонён из осторожности (Alakazam уже на потолке 93%+, размен невыгоден независимо от точных чисел).
      Взамен: чисто АДДИТИВНЫЙ патч — Mega Abomasnow ex matchup detection (карты 722/723, официальный
      starter-kit конкурса, крупный field-кластер, ранее падал в generic=220). 0 exc/ill, presubmit PASS.
      Память: [[project-v034-v036-study-and-v044]]. Юзер подтвердил («Ок») — залито.
- [x] **J-fix-N8 (v045_archaludon_judge_fix) ГОТОВ, НЕ засабмичен.** Юзер сообщил о 3 живых проигрышах подряд
      топ-100 Alakazam-пилотам (Julian rank71, Bulba-Zero rank80, Aikimi rank107) на v044, попросил разобрать
      реплеи. Скачано через `kaggle competitions episodes/replay` (CLI list/download РАБОТАЕТ несмотря на
      сломанный KGAT для submit). **Находка:** игры кончались на ходу 4-9 при нашем полном HP — рука
      оппонента росла **12→21→23 карты за 2-3 хода** (Powerful Hand ~20 dmg/карту = 460+ урона, безусловный
      снос). В 2 из 3 проигрышей у нас на руке лежал **Judge (1213, уже в деке)** несколько ходов подряд, пока
      рука оппонента росла на глазах — но card_id==1213 НЕ ИМЕЛ scoring-логики вообще (падал в generic 1000).
      **Фикс:** Judge получает приоритет 19000 (выше Boss/Lillie/Explorer) при `matchup==alakazam` и
      `opp_hand>=8` — используя уже существующую `_estimate_alakazam` модель. Без изменения деки. Локальный
      гейт N=150: v017(Alakazam-прокси) 0.933→0.953(v044)→**0.987**(v045), v009a/v007/v033 не просели (0.853/
      0.803/0.813). 0 exceptions, presubmit PASS. Это принципиально ДРУГОЙ механизм чем закрытый 4 раза
      reactive-retreat (hand-disruption через уже имеющуюся карту, не позиционирование). Память:
      [[project-v045-judge-alakazam-forensic]]. НЕ сабмитить без явного «ок» юзера (v044 уже в tracked-2,
      v045 вытеснит v043 или займёт второй слот).
- [x] **J-fix-N9 (v045, доп. Dipplin-фикс) ГОТОВ.** Юзер прислал ссылку на живой проигрыш v044 против Dipplin
      ("практически ваншотает") — episode 86703807. Скачан и разобран: дека оппонента = Dipplin/Swirlix/
      Festival Grounds + **Brave Bangle x4** (+30 dmg vs {ex}). Атака "Do the Wave" = **20×бенч АТАКУЮЩЕГО**
      (не наш!), бенч оппонента был полон (5) весь матч → 100 база +30 Bangle = 130/удар, ×2 через Festival
      Lead (атака дважды, и снова если первая нокаутит) = 260 — снесло Archaludon ex с 200/400 HP за один ход.
      Старый `opp_max_damage()` для "dipplin" был статичной оценкой (200/400), не учитывал ни реальный бенч,
      ни Brave Bangle. **Фикс:** новая `_estimate_dipplin(obs)` — `20 * len(opp_bench_pokemon(obs))` +30 если
      наш активный ex и виден Brave Bangle, ×2 если Festival Grounds в игре — та же архитектура что
      `_estimate_alakazam`. Юнит-тест на синтетическом obs подтвердил (110,220) корректно. Построен тестовый
      фикстур `submissions/_dipplin_test_opp/` (дека из реплея + generic-greedy пилот, НЕ копия реального
      игрока) — 0 exceptions в regression-прогоне, полный пул (v017/v018/v009a/v036) тоже без регрессий.
      Память: [[project-v045-dipplin-forensic]].
- [x] **J-fix-N10 (v045) ЗАЛИТ 2026-07-18.** Свежий единый прогон v036/v044/v045 против одного пула
      (v017/v018_grimmsnarl/v035_makthanithin/v009a/v007/v033, N=150 каждый) показал все три версии
      статистически неразличимы (разброс 2-8пп, шумовой пол) — локальный гейт не может подтвердить/
      опровергнуть Judge/Dipplin-патчи (ни один оппонент не воспроизводит live-механизм), только "ничего не
      сломал" (0 exceptions везде). Юзер подтвердил («Отправляй последнюю версию») — залито. Submission
      **54810464 PENDING**. tracked-2 стало {v045_archaludon_judge_fix PENDING, v044_archaludon_abomasnow
      899.8} — v043_alakazam_mirror_fix (506.9) вытеснен.

**Не делать (гвардрейлы):**
- НЕ сабмитить v039 без явного «ок» юзера ([[feedback-language-and-submit]]).
- НЕ запускать PPO retrain без подтверждения (GPU-прогон + «ок»).
- НЕ менять main.py v039 (чекпойнт v038 намеренно оставлен — deck-swap эксперимент; смена чекпойнта
  только после J-fix-N2).
- Критерий «дека сломана» (31 энергия) — объективный; но «своп deck.csv в одиночку = 20:19»
  ([[project-deck-policy-entanglement]]) — поэтому J-fix-N1 может показать малую дельту, и это НЕ повод
  отказываться от fine-tune (J-fix-N2).

---

## K — PLAN: Alakazam counter (started 2026-07-19, после первых 2 live-проигрышей v046_final)

**Триггер.** v046_final (submissionId 54815255, залит 2026-07-18 20:01) — первые 4 публичные партии:
2 победы, **2 поражения, обе от Alakazam** (higaki ep86751133, daiki_H ep86751593). Разобрано вручную
через `kaggle competitions replay <id>` (открыт для любых эпизодов, см. [[project_episode_mining]]).

**Подтверждённый механизм (арифметика сходится, не гипотеза):** Alakazam attack "Powerful Hand" кладёт
**прямые damage counters = 2 × (карт в руке атакующего)** — это НЕ "damage" в смысле effective_damage()
(обходит Full Metal Lab/resistance/HP-стены). higaki: 23 карты → 460 урона (наш Archaludon ex 300 HP).
daiki_H: 21 карта → 420 урона (даже усиленный до 400 HP Hero's Cape'ом Archaludon ex не пережил).
Это тот же механизм, что и в [[project_v033_alakazam_mechanism_closed]] (Powerful Hand OHKO), где
**reactive-retreat закрыт 4 раза подряд (v012/v017/v036×3/v033)** — новую правку по этому направлению
(вариации retreat/threshold) не делать, нужен принципиально другой рычаг.

**★Новая находка (уточняет [[project_v045_judge_alakazam_forensic]]):** Judge (v046 x2 копии, скорингtriggers
на `opp_hand>=8` при matchup==alakazam, main.py `score_play()` ~строка 851) **пришёл в руку рано в ОБЕИХ
проигранных партиях** (higaki: ход 2-3; daiki_H: ход 2, 4) и **ни разу не был разыгран** — исчез из руки
в тот же ход (в сброс) без единого триггера эффекта, порог `opp_hand>=8` ещё не был достигнут (рука
оппонента на тех ходах = 4-5 карт). Вторая копия за всю партию так и не была замечена в руке ни разу ни
в одной из двух партий. Подробности: [[project_v046_judge_wasted_early_finding]].

**Форум-рекон 2026-07-19 (`kaggle competitions topics list/show`, прямой доступ, PATH нужен `.venv/bin`
впереди системного, скрипт из kaggle-skill плагина сконструирован под другую версию CLI — работает
голый `kaggle competitions topics show <id>`):**
- **Топик 716207** (Sanjay20052) — Cornerstone Mask Ogerpon ex (Ability-иммунитет) новый нативный контр
  Archaludon в поле. Не про Alakazam, для общей осведомлённости о мете, действий не требует сейчас.
- **★Топик 712811** (Articuno0144, баг-репорт, 0 ответов от организатора) — **Unfair Stamp на движке
  работает НЕ как написано на карте**: добирает карты ДО 5/2 без предварительного возврата руки в
  колоду ("both players draw cards until they have 5/2 cards... hand is not returned to the deck")
  → карта "стала очень слабой". Это прямо бьёт по идее naoto714 (см. ниже) — ПЕРЕД тем как доверять
  любой WR-цифре с Unfair Stamp, надо перепроверить поведение на НАШЕМ движке (может отличаться —
  движок мог быть пропатчен с 06-23, или у автора то же самое было и его +23.9пп уже это учитывает).
- **★★★Топик 713608** (david valor, 25 голосов, детальный технический пост-мортем) — не про Alakazam
  напрямую (это у них СВОЯ дека Starmie против Lucario), но методология прямо применима к нам:
  - **"Forensic loss-analysis trap" (§3.5):** они прочитали 40 проигранных партий, LLM нашёл 3
    убедительных фикса, заимплементили, **paired A/B дал −7.6пп регресс** (p=0.003) — "выборочная
    слепота": чтение только поражений скрывает игры, выигранные тем же самым паттерном поведения.
    **Мы сейчас в точности в этой ситуации** (прочитали 2 живые партии, нашли механизм, готовы
    патчить) — НЕ шипить код-фикс без честного paired A/B, N≥400.
  - **Eval-дисциплина (§4):** "77% на 200 играх стало 67% на 400" — ревалидировать любой перспективный
    результат на N≥400 прежде чем доверять. Hill-climbing ниже шумового пола — реальная ловушка
    (округляли 0.786→0.921 по 0.5пп шагам, честная цифра была 0.88, не 0.92).
  - **Комментарий Chrismaghuhn в топике 721338** ("How do you teach an agent when to use Boss's
    Orders/Ultra Ball"): "for a search-based agent you usually don't teach the triggers at all — you
    let the search evaluate the card as one branch and price it by outcome." Это ровно то "принципиально
    другое" направление, которое требует [[project_v033_alakazam_mechanism_closed]] взамен очередной
    heuristic-вариации — у нас уже есть PIMC v2 (`src/pilot/pimc_v2.py`), не проверяли его именно на
    выборе supporter-карты (Judge/Boss/Explorer/Lillie) конкретно в Alakazam-матчапе.

### K1 — Проверить движок: реально ли Unfair Stamp сломан у нас (дёшево, сначала)
Изолированный скрипт на cg_native: разыграть Unfair Stamp, сравнить размер руки до/после с ожиданием
по тексту карты (обе руки → 5/2 БЕЗ возврата старой руки в колоду = баг; С возвратом = работает).
Если баг подтверждён — Unfair Stamp/Xerosic откладываются, не тратить сабмит-слот на мёртвую карту.

### K2 — Починить конкретный баг «Judge тратится впустую» (узкий, низкий риск)
Найти точный код-путь, который выбирает Judge как «топливо» для Ultra Ball/другого cost-эффекта
(или иной сброс), пока matchup==alakazam и Judge ещё не разыгран — защитить карту от такого сброса
явным правилом. Это не смена порога, а защита уже принятого решения «эту карту решили придержать»
от случайной потери. Гейтить локально (N≥400 vs v017_alakazam_heur + реальные Alakazam-оппоненты
пула) до рассмотрения сабмита.

### K3 — Пересмотреть сам триггер `opp_hand>=8`
Честный paired A/B (N≥400): «играть Judge на первой же возможности при подтверждённом matchup==alakazam»
(без ожидания порога) против текущего порога. Обоснование: Judge — ресурс «используй или потеряешь»,
и ожидание уже дважды стоило нам карты целиком. Не полагаться на интуицию — измерить обе версии.

### K4 — Deck-slot эксперимент с hand-disruption (Unfair Stamp/Xerosic), ТОЛЬКО если K1 подтвердит рабочесть
Если карта реально работает на нашем движке: повторить ту же 4-way абляцию, что уже использовалась для
Hero's Cape/Judge решения (deck-only/logic-only/combo/baseline, N=200-400/ячейка) — НЕ слепой своп.
Hero's Cape резать НЕЛЬЗЯ (уже доказано: −8.9…−10.5пп vs Crustle, см. [[project_v046_final_gated_not_shipped]]).
Кандидаты на срез тоньше некуда (Full Metal Lab уже урезан под 2-ю Judge) — нужен реальный анализ слотов
колоды, не гадание.

### K5 (больше ставка, опционально, в конце очереди) — PIMC-driven supporter-выбор
Проверить, обгоняет ли неглубокий PIMC/search-driven выбор между Judge/Boss/Explorer/Lillie (оценка
веткой по исходу rollout) наш hand-tuned priority-порядок — это и есть «принципиально другой механизм»
по формулировке [[project_v033_alakazam_mechanism_closed]]. Наивысшая неопределённость и стоимость —
делать последним, только если K2/K3 не закрывают разрыв.

**Eval-дисциплина по всем пунктам (напрямую из david valor §4, соблюдать строго):**
N≥400 на сравнение, ревалидировать любой перспективный N=200 результат; Wilson CI, не точечная оценка;
**не шипить код-фикс по прочтению нескольких проигранных реплеев без честного paired A/B** — форензик
реплеев даёт ГИПОТЕЗЫ, разрешение на код-чендж даёт только гейт (или live-probe, если известно что
локальный гейт — слабый прокси именно для Alakazam, см. generalization gap
[[project_v033_archetype_diversity_confirmed]]).

**Не делать:** новые вариации reactive-retreat (закрыто 4 раза); слепой своп Hero's Cape (доказанный
регресс vs Crustle); доверять WR-цифрам с Unfair Stamp до K1.

### K1 РЕЗУЛЬТАТ (2026-07-19): Unfair Stamp НЕ бажен, но НЕСОВМЕСТИМ с Hero's Cape (оба ACE SPEC)
Проверено эмпирически на `src/cg_native` (macOS native engine, `battle_start` с двумя ACE SPEC в одной
колоде) — **errorPlayer=0, errorType=4** (та же ошибка легальности, что и у "Hero's Cape >1 копия"),
т.е. движок реально применяет диковый лимит "1 ACE SPEC суммарно на колоду", а не только "1 копия
каждой карты". Наша колода уже содержит Hero's Cape (ACE SPEC) → Unfair Stamp физически нельзя
добавить БЕЗ среза Hero's Cape целиком, а срез Hero's Cape уже доказанно даёт −8.9…−10.5пп vs Crustle.
Отдельно проверен исходник движка (`data/engine/ptcg_src/CardImpl.h:3289`) — реализация Unfair Stamp
корректна (shuffle-then-draw-5/2, ровно по тексту карты), баг из топика 712811 (06-23) на текущей
сборке не воспроизводится (либо был починен позже, либо специфичен другой версии).
**★Найдена лучшая альтернатива при чтении того же исходника:** **Xerosic's Machinations (1197,
Supporter, CardImpl.h:3940)** — "opponent discards to 3 cards", условие `opp_hand>=4` (значительно
раньше нашего текущего Judge-порога 8), **НЕ ACE SPEC** (не конфликтует с Hero's Cape), бьёт только
руку ОППОНЕНТА (не трогает нашу). Ирония: именно эту карту сыграл ПРОТИВ нас higaki в ep86751133 (см.
K2). Также найден **Hand Trimmer (1087, Item, не ACE SPEC)** — оба игрока сбрасывают до 5, не требует
supporter-слота. Оба — сильные кандидаты для K4 вместо Unfair Stamp; статус K4 не начат.

### K2 РЕЗУЛЬТАТ (2026-07-19): фикс реализован и прогейчен, НЕ засабмичен
Разобрав episode 86751133 глубже (сигнал: переход состава руки + `supporterPlayed`, не только `logs`
— **более раннее заявление "Judge вообще не был сыгран в обеих партиях" ОШИБОЧНО**, см. исправление в
[[project_v046_judge_wasted_early_finding]]): на ходу 3 higaki сыграл против нас Xerosic's Machinations
(1197), заставив сбросить руку [190×3, 1185, 8, 1213] до 3 карт. `score_discard()` в
`submissions/v046_final/main.py` не имел спец-обработки ни для JUDGE (падал в generic 1000), ни лимита
копий для ARCHALUDON_EX (флэт `-5000 keep` на КАЖДУЮ из 3 копий в руке) — агент сохранил все 3
избыточные копии Archaludon ex и сбросил Judge/Explorer's Guidance/Metal Energy. В daiki_H (86751593)
Judge наоборот сработал штатно (opp hand 10→4 на ходу 4), просто одного сброса не хватило — рука
отросла обратно до 21 карты за 13 ходов, вторая копия Judge за игру не пришла — это отдельный, уже
известный предел ("one copy isn't enough"), не тот же баг.

**Правка** (`submissions/v046_judgefix/`, копия v046_final с 2 узкими изменениями в `score_discard()`):
(1) JUDGE получает `-4000 keep` при matchup==alakazam, работает при ЛЮБОМ forced-discard контексте, не
только Ultra Ball; (2) ARCHALUDON_EX "keep"-бонус капается по числу копий в руке (`-5000` только 1-й
копии, `-1500` любой 2+й) — 3 идентичные копии одной эволюционной ступени в руке не нужны одновременно.
`v046_final` НЕ тронут (восстановлен в исходное состояние — эта директория совпадает с уже залитым
submissionId 54815255, живой сабмит не трогаем).

**Локальный гейт** (`scripts/v046_judgefix_gate.py`, native engine, N=200×3 оппонента, seed 20260719):
v017_alakazam_heur delta=+0.000 (98.0% оба — прокси не воспроизводит именно этот механизм), v009a_crustle
delta=−2.1пп (p=0.545, шум), v018_grimmsnarl delta=−1.5пп (p=0.398, шум). **Crustle — исторически хрупкий
матчап (Hero's Cape-регресс), поэтому по дисциплине david valor (§4, "revalidate at N≥400") перепрогнан
отдельно на N=400 (seed 20260720): delta развернулся в +1.0пп (p=0.703)** — подтверждает, что −2.1пп на
N=200 был чистым шумом ровно как предупреждал форум-пост. **Вывод: 0 регрессий за пределами шума на всех
трёх проверенных матчапах.** Локальный гейт не может подтвердить УЛУЧШЕНИЕ (v017-прокси не реплицирует
Xerosic-механизм из живой партии) — как и раньше для Judge/Dipplin-патчей, только "ничего не сломал".

**Статус (ИСПРАВЛЕНО): ЗАСАБМИЧЕН 2026-07-19 05:04, submissionId 54822473** — текст выше был написан до
сабмита и устарел. v046_judgefix живёт в tracked-2 вместе с v044_resub; live-результат см. §L ниже.

---

## L — FORENSIC: полный разбор 129 live-игр v044_resub/v046_judgefix → кандидаты v048 (2026-07-20)

> **Контекст:** v044_resub (N=68, WR 66.2%) и v046_judgefix (N=61, WR 50.8%) — оба сейчас в tracked-2.
> Прошлый матчап-разбор (session-memory, N=56/63) оставлял ~40% игр в бакете "generic/other" —
> слишком грубо для выводов. Полный форензик всех 129 игр (не выборка, и явно не только поражения —
> см. david valor "forensic loss-analysis trap", топик 713608) — методология, таблицы и разбор
> кейсов: **[docs/gold/v047_v048_forensic.md](docs/gold/v047_v048_forensic.md)**. Пайплайн:
> `scripts/forensic_v044_v046_full.py` → `data/episodes/v044_v046_forensic.csv` (129 строк) →
> `scripts/analyze_v044_v046_forensic.py` → `data/episodes/v044_v046_forensic_summary.md` (полные
> W-vs-L кросс-таблицы). Каждая находка обязательно сверена между победами и поражениями, иначе не
> считается сигналом.

**Ключевые находки** (детали и таблицы — в docs/gold-файле):
- Расширение архетип-сигнатур двумя новыми кластерами (archaludon_mirror N=19, dragapult N=14 —
  крупнее половины уже существующих в `detect_matchup()`) сократило "other" с 37% до 12% игр. **Ни
  одной из этих веток нет в живом `detect_matchup()`** — 26% живого поля (33/129) сейчас получает
  вообще никакой matchup-специфичной логики, не по причине сложности матчапа, а по причине пробела
  в покрытии.
- С исправленной разбивкой регрессия v046_judgefix оказалась ШИРЕ, чем считалось: не только
  generic/lucario (известно раньше), но и archaludon_mirror (66.7%→42.9%) и dragapult (100%→50%) —
  указывает на причину уровня ДЕКИ (влияет на все матчапы одинаково), а не логики конкретного матчапа.
- `bench_thin_frac` — валидированный (не шум) W-vs-L сигнал в обоих агентах, и v046 систематически
  тоньше v044 в ОБЕИХ группах (W и L) — ещё один довод в пользу deck-level причины (Full Metal
  Lab→2-я Judge срез), не code-фикса.
- Boss's Orders play-rate у v046 в поражениях (16.7%) вдвое ниже, чем в его же победах (38.7%) и чем
  у v044 в любой группе (33-39%) — согласуется с гипотезой "меньше ресурсов" от среза FML.
- Задержка распознавания матчапа (`detect_lag_turn`) НЕ отличает победы от поражений ни у одного
  агента — механизм, который чинит v047_deckbelief (persistence detect_matchup), не показывает
  видимого WR-сигнала в этих данных (не отменяет v047 гейт-результаты, просто не переоценивать этот
  рычаг конкретно для разрыва v044→v046).

### ★★★ N=1000 ПЕРЕГЕЙТ (2026-07-20) — v047_deckbelief ПЕРЕСМОТРЕН, это РЕГРЕССИЯ vs v044, не паритет

`scripts/deckbelief_gate_v047.py --games 1000 --h2h-games 1000 --opponents v007_crustle,v009a_crustle,
v017_alakazam_heur,v018_grimmsnarl,v024_cardemb_rl,v033_plan_d_ppo,v034_romanrozen_clone,
v036_archaludon_patched,v039_ft,v046_judgefix` (21000 игр, 0 exceptions/aborts, 464s,
`results/gates/v047_full_pool_gate_1000.log`). Прошлый гейт (N=200-300, тот же script с 3-опп пулом)
показал паритет (h2h WR=0.470 [0.414,0.527]) и "0 регрессий" — на N=1000 картина другая:

- **Прямое зеркало v047 vs v044: WR=0.447 [0.416, 0.478]** — верхняя граница ДИ ниже 0.5, статистически
  значимо ХУЖЕ v044, не паритет.
- **3 из 10 оппонентов пула — статзначимая регрессия** (p<0.05): v036_archaludon_patched −5.1пп
  (p=0.022), v039_ft −4.8пп (p=0.028), **v046_judgefix −4.8пп (p=0.032)**.
- Ещё 3 отрицательные, но не значимые: v009a_crustle −0.5пп, v024_cardemb_rl −1.1пп,
  v034_romanrozen_clone −3.0пп. Итого **6 из 10 delta отрицательные**, ни одна из 4 положительных
  дельт не значима даже близко (ближайшая v017_alakazam_heur p=0.058).

Ровно тот паттерн, о котором предупреждает david valor (§4 outcome, "77% на 200 играх стало 67% на
400") — прежний вывод "0 регрессий, паритет" был статистически недостаточно обеспечен (N=200-300 на
3-опп пуле), реальная картина на N=1000×10 — чистая регрессия. **L1 ниже (база v048 = v047_deckbelief)
недействителен как написан** — строить v048 на v047 сейчас значит наследовать эту регрессию.
Причина регрессии НЕ исследована (кандидаты: belief-net численные ветки ошибочно триггерятся на
матчапах вне тренированных прототипов, либо persistent-детекция иногда переопределяет ВЕРНОЕ
board-only "generic"-чтение неверным belief-guess) — до объяснения причины не пытаться чинить вслепую.

**Приоритизированные кандидаты для v048** (каждый требует отдельного honest paired A/B, N≥400 —
форензик даёт гипотезу, не разрешение на код-чендж, дисциплина §K сохраняется):

### L0 (НОВЫЙ, теперь приоритет №1) — Диагностировать причину N=1000-регрессии v047 vs v044
До любого дальнейшего использования v047_deckbelief как базы: собрать trace/detail (belief-net
guess vs board-only guess vs правда архетипа) конкретно на v036/v039_ft/v046_judgefix матчапах —
это единственные 3 с доказанной регрессией, там и искать механизм. Если причина в persistent-детекции
(переопределяет верный board-only guess неверным) — рассмотреть более консервативный merge (belief-net
только когда board-only вернул "generic", никогда не переопределяет уверенный board-сигнал). Это
предшествует L1, не отменяет его — L1 остаётся правильной идеей (изолировать code-diff от deck-diff),
но применительно к ИСПРАВЛЕННОЙ версии v047, не к текущей.

### L1 (ОТЛОЖЕН до L0) — База v048 = v047_deckbelief + узкий порт score_discard() из v046_judgefix, БЕЗ среза FML
Порт ТОЛЬКО code-diff (JUDGE `-4000 keep` protection + Archaludon ex per-copy cap), deck.csv остаётся
как у v047 (= v044, БЕЗ 2-й Judge / без среза Full Metal Lab). Гипотеза: crustle-фикс (14.3%→55-62%)
переживёт порт, а generic/lucario/mirror/dragapult-регрессия — нет, т.к. она из deck-уровня. Нужен
4-way ablation-гейт (deck-only/logic-only/combo/baseline, N≥400/ячейка, новый
`scripts/v048_judgefix_port_gate.py`) — самый высокий приоритет, самая сильная поддержка данными.

### L2 — Добавить archaludon_mirror/dragapult ветки в detect_matchup()
`{169,190,666}`/`{119,120,121}`. Дёшево, механически идентично уже существующим 8 веткам, закрывает
26%-дыру в покрытии. Собственный узкий гейт, низкий риск.

### L3 (эксплораторно, НЕ прямой порт) — Митигация тонкой скамейки
Корреляция с поражением подтверждена (2.6-3.2× на `bench_thin_frac==1.0`), но ручное чтение показало
неоднородный механизм (симптом отставания в размене в одном кейсе, вероятно движковый edge-case в
другом — см. docs/gold-файл) — слепое правило "держи bench≥2" не гарантированно безопасно. Переоценить
ПОСЛЕ гейта L1 — регрессия может исчезнуть сама, если гипотеза deck-уровня подтвердится.

### L4 (низкий приоритет, не цель v048) — Аномалия "зависшей" короткой игры
episode 86933558: 69 шагов, ход застрял на 4, повторяющиеся почти идентичные select-запросы, обрыв без
резолюции. Похоже на движковый/replay-export edge case, не на баг агента — точечно перепроверить на
нескольких похожих коротких (<80 шагов) поражениях, не приоритет.

**Не делать повторно** (уже закрыто предыдущими прогонами, этот форензик не даёт новых оснований):
K4 hand-disruption для нашей колоды (офиц. датасет N=2416: −16.1/−10.4пп), K5 PIMC-supporter (N=400
отрицательно на всех 3 матчапах), reactive-retreat (закрыт 4×), срез Hero's Cape (доказанный регресс
vs Crustle). Находка про Xerosic оппонента ПРОТИВ нас (v044: 34.8% в поражениях vs 15.6% в победах) —
зеркальный аргумент: защищать свои ресурсы (Judge-protection, уже в v046), не добавлять disruption

### ★★★★★ ИТОГ ночной сессии 2026-07-20/21: L0 диагностирован, L2 реализован, L1 4-way гейт ПРОЙДЕН, кандидат v048 собран (НЕ засабмичен)

**v047_deckbelief ЗАСАБМИЧЕН 2026-07-20** (юзер явно одобрил после того как узнал о N=1000-регрессии —
осознанное решение, "всё равно 44 версия больше не проверяется на кагле"; tracked-2 теперь
{v047_deckbelief, ...}). Диагностика ниже велась параллельно/после, локально, без дальнейших сабмитов.

**L0** (`scripts/deckbelief_l0_diagnose.py`, N=200×3 доказанно-регрессировавших оппонента):
уточнил гипотезу из §L0 выше — persistent-детекция НЕ переопределяет верный board-only guess (board
безусловно побеждает в коде), реальный механизм у́же: против archaludon_mirror-оппонентов
(v036/v046_judgefix, оба реально archaludon_mirror по деке) board-скан не покрывает LINE {169,190,666}
→ belief-net вступает почти всегда и в ~44% игр ошибочно навешивает "crustle" (потребляемый downstream:
opp_max_damage=120, apply_overrides Crustle-ветка) вместо нейтрального "generic". НО: частота
симметрична между победами/поражениями (43.5% win/44.4% loss у v036) — по форензик-дисциплине НЕ
доказанный причинно-следственный сигнал сам по себе. Для v039_ft (true=crustle, тоже один из 3
значимо-регрессировавших) misclassification почти не срабатывает (1.6%/0%) — отдельный
`scripts/deckbelief_l0_v039_dig.py` (N=300) проверил и альтернативную гипотезу (persisted stale
"crustle"-лейбл вместо live board-generic) — тоже симметрично (15.3% win/16.9% loss) — причина
регрессии v039_ft осталась недиагностированной, но при этом решено не чинить вслепую, а положиться на
L1-гейт (ниже он оказался чист и для v039_ft).

**L2** (реализован в `submissions/v048_base/main.py`): добавлены `ARCHALUDON_MIRROR_LINE={169,190,666}`
и `DRAGAPULT_LINE={119,120,121}` в `_detect_matchup_board_only()` — чисто структурная правка (board
теперь ловит эти архетипы напрямую, belief-net для них почти перестаёт вызываться). Числовые ветки
(`opp_max_damage` и т.п.) НЕ добавлены — сверка по `data/EN_Card_Data.csv` показала, что реальный урон
Archaludon ex Metal Defender (220) и Dragapult ex Phantom Dive (200) уже близок к существующему
безветочному дефолту (220), инвестировать в неподтверждённую точную цифру признано лишним. Проверка на
N=200×2: ACTIONABLE-wrong rate у belief-net упал с ~44% до ~4-10%.

**L1** (`scripts/v048_judgefix_port_gate.py`, N=400/ячейка, полный 10-опп пул + h2h vs v044, 21600 игр,
0 aborted/exceptions): 4-way ablation на базе L2-исправленного v048_base —

| Вариант | deck.csv | score_discard() | Регрессия vs v044 (10-опп пул) | h2h vs v044 |
|---|---|---|---|---|
| `v048_baseline` | v044 | v044 | НЕТ (0/10) | 0.482 |
| `v048_deck_only` | v046_judgefix (FML→2×Judge) | v044 | НЕТ (0/10) | 0.480 |
| `v048_logic_only` (буквальная рекомендация L1 выше) | v044 | +Judge-protect+Archaludon-cap | **ЕСТЬ** (v024_cardemb_rl −4.0пп p=0.027) | 0.468 |
| `v048_combo` (= полный диф v046_judgefix НА L2-базе) | v046_judgefix | +Judge-protect+Archaludon-cap | НЕТ (0/10) | 0.480 |

Главный вывод: **сам L2-фикс (покрытие archaludon_mirror/dragapult) уже устраняет N=1000-регрессию
v047** — `v048_baseline` (просто v047+L2, без единого дальнейшего изменения) чист на всём пуле, включая
ровно те 3 оппонента (v036/v039_ft/v046_judgefix), что были значимо хуже у голого v047. Это меняет
интерпретацию §L1 выше: буквальный "logic-only без deck-среза" (то, что предполагалось самым безопасным)
оказался ЕДИНСТВЕННЫМ вариантом с обнаруженной регрессией, а `v048_combo` (полный диф v046_judgefix,
включая deck-срез, но воспроизведённый НА исправленной L2-базе) — чист И даёт лучший (хоть и не строго
значимый, p=0.09) тренд по v009a_crustle (+4.0пп) среди всех 4 вариантов, а archaludon_mirror/dragapult
дельты у него неотрицательны (v036 +2.7пп, v046_judgefix +2.3пп) — то есть deck-срез сам по себе
оказался безопасен именно на L2-исправленной базе, живая широкая регрессия v046_judgefix была
специфична для его ИСХОДНОГО plain board-only detect_matchup(), не для среза как такового.

**Кандидат**: `submissions/v048_deckbelief_judgefix/` = копия `v048_combo`. Honest exec-smoke
(Docker `pokemon-tcg-sim-torch`, реальный linux x86_64 движок, `exec()` без `__file__` — тот же класс
проверки что ловил 3 живых бага у v047) — PASS, 0 exceptions, belief_model_loaded=True,
matchup=archaludon_mirror корректно распознан в реальном прогоне. **ЗАСАБМИЧЕН 2026-07-20** (юзер явно
одобрил "засылай" после обзора плана/результатов гейта) — tracked-2 стало
{v048_deckbelief_judgefix, v047_deckbelief}. Live-μ TBD, ждать 150-250 партий.

Рабочие директории `submissions/v048_base|baseline|deck_only|logic_only|combo/` оставлены как есть —
провенанс ablation-таблицы выше, не мусор.
своей стороны.

---

## M — PLAN v6: BAND-GAUNTLET → v072_mirror_gauntlet (2026-07-24, ⏳ PENDING USER APPROVAL)

> **Цель юзера: μ1050+ (верх серебра — золото).** Пороги 07-24 (5619 команд): бронза 851 (top-562),
> серебро 920 (top-281), золото 1088 (top-21). Мы: место 1583, μ758.
> **Доказательная база:** `docs/gold/top_ladder_recon_20260724.md` (1273 игры топ-5 за 07-23) +
> live-вердикт v070+P0 (107 игр, память `project_v070_p0_live_verdict_20260724`) +
> band-математика (myso1987 07-22 × наши live-профили).

### Ключевые факты, на которых стоит план

1. **4 из 5 топ-команд (μ1123-1192) играют НАШ jiatu.l-лист Grimmsnarl** (±2 тех-карты). Дека закрыта.
2. **Все топ-5 — детерминированные эвристики/сошедшиеся BC, НЕ search.** Топ-1 выигрывает всего 56%.
   Рейтинг = равновесие WR vs свой бэнд; для 1050 нужно ~52-55% на бэндах 900-1100.
3. **Наш блокер — болото 700-899**, не потолок наверху: Lucario 26% поля на 700-799 (наш WR 29%),
   Archaludon 29% на 800-899 (наш WR 21%). Ожидаемый WR v070+P0 на этих бэндах 44%. Наверху эти
   архетипы почти вымерли; зеркало Grimmsnarl = 40-48% игр топ-бэнда, Alakazam = 20-28%.
4. **BC-приор больше не тянет:** v070-гибрид = уровень автора (750); точечные override поверх BC
   не двигают WR (P0: 4 срабатывания на 107 игр). Поднимает только качество плейбука.

### M0 — База и рамки

- **База пилота:** `v065_defensive_swap` (последний heuristic-чемпион линии: v064 БАГ-фиксы + P0 своп
  «копия переживает OHKO» + P2 Boss-guard). Дека v064/jiatu.l — **без изменений** (миграция теха — §M5, отдельно).
- **Имя версии:** `v072_mirror_gauntlet` (v071 занят локально-отрицательным router-экспериментом).
- Per-decision stderr-логи обязательны (feedback_always_agent_decision_logging), exec-smoke под
  linux/amd64 exec() без `__file__` обязателен.

### M1 — Зеркальный плейбук (5 механик топов; зеркало = 40-48% игр наверху, у нас live 45%)

- [ ] M1a **ex-ротация ПЕРЕД атакой** (расширение v065-свопа): активная Grimmsnarl ex повреждена
      (HP < порога размена) И на скамейке здоровая заряженная копия → ротация ДО атаки, не только
      под смертельный OHKO на матчпойнте. У Luca 150 таких свопов/день, 71% — в зеркалах.
- [ ] M1b **Adrena-Brain каждый доступный ход** — проверить v062-логику на пропуски; целевая метрика
      ≥4-5 активаций/игру (экспертный уровень 4.2-5.2). Chip решает sudden-death в зеркале.
- [ ] M1c **Splash-double-prize**: таргетинг с учётом +30 splash Shadow Bullet — предпочитать
      состояния «у оппа бенчер ≤30hp» (двойной приз); симметрично — лечить/убирать СВОИХ бенчеров ≤30hp.
- [ ] M1d **Boss-таблица зеркала**: чужой Munkidori / слабый не-ex бэклайнер (в v062 уже есть —
      верифицировать live-срабатывание по логам).
- [ ] M1e **Rare Candy темп**: кэнди→ex первым же ходом, когда доступно (проверить приоритеты).

### M2 — Lucario recovery (болото 700-799: 26% поля)

- [ ] Diff v063→v064 lucario-веток: v063 live 60% vs Lucario, v064 сломал до 39% (размен со Starmie-фиксом).
      Форензик 17 lucario-поражений v064 (реплеи в `out/v064_full_download/`) → восстановить v063-поведение,
      сохранив Starmie-фикс (78%).
- Acceptance: vs nursrijan + maulikgajera Lucario ≥55% (N≥1000 суммарно), Starmie-прокси без регресса.

### M3 — Anti-Archaludon экспертный рецепт (болото 800-899: 29% поля)

- [ ] Ранний фокус Duraludon ДО эволюции (Boss + splash + Adrena-роутинг в резервные Duraludon);
      не придерживать Shadow Bullet из-за встречного урона; M1a покрывает ротацию танков.
      Рецепт из `data/il_grimmsnarl_v2/archaludon_recon/` (реконструкции экспертных побед).
- Acceptance: vs v036/v044-класс ≥40% (реалистично: стена реальна, 25%→40% и есть цель).

### M4 — Xerosic-тайминг vs Alakazam (верх: Alakazam 37-39% бэндов 900-1099)

- [ ] Играть Xerosic's Machinations в окно ПЕРЕД ожидаемым Powerful Hand (opp_hand ≥ 8 и наш ход
      перед их атакой), как Majkel1337 (#4). Проверить текущий триггер, убрать раннее сжигание.
- Acceptance: vs prvsiyan Alakazam v8/v9 без регресса, в идеале +3-5пп (у нас live уже 67% на низком бэнде).

### M5 — (опция, ПОСЛЕ v072-вердикта) Тех A/B: +2 Handheld Fan

- Зеркальный контр-тех топ-1: 75% зеркала vs 56% без него; цена — просадка vs Alakazam/Crustle.
  Отдельным сабмит-слотом, только после live-вердикта v072 и только с явного «ок».

### M6 — Гейт (новый: по бэндам, не «среднее по больнице»)

- [ ] `scripts/gate_band_weighted.py`: три бэнд-микса из myso1987 07-22 —
      700-799 {lucario .26, archaludon .18, alakazam .17, dragapult .12, rest},
      800-899 {archaludon .29, alakazam .27, lucario .07, crustle .05, mirror .05, rest},
      900-1099 {alakazam .38, mirror .23, crustle .09, archaludon .09, lucario .07, rest};
      оппоненты из имеющихся: nursrijan/maulikgajera (lucario), v036/v044 (archaludon),
      prvsiyan v8/v9 + v016/v017 (alakazam), v062/v063/v070-гибрид (mirror), v009a/tusk_crustle (crustle).
- **Acceptance (все обязательны):** (1) взвешенный WR ≥52% на КАЖДОМ из трёх бэнд-миксов;
      (2) зеркальный h2h ≥55% vs v070-гибрид И vs v063; (3) 0 значимых регрессий vs v065 по полному
      CORE_PAIRS-пулу N=1000; (4) 0 исключений; (5) Docker exec-smoke PASS.
- [ ] Фулл-прогон N=1000 (конфиг `project_full_gate_pool_config`) + M1b-метрика Adrena/игру в логах.

### M7 — Сабмит и оценка

- [ ] Двойной идентичный сабмит v072 (поручение юзера 2026-07-24, после его апрува плана и PASS §M6)
      → tracked-2 = {v072A, v072B}, вытесняет обе копии v070+P0.
- [ ] Live-оценка по `latest-matchup-wr` через 12-24ч (50-70 игр = грубая оценка), полный вердикт 150+ игр.
- Fallback: если v072 live < v070-уровня (~750) после 150 игр — ресаб v044 (страховка бронзы, 891 > порога 851)
  отдельным решением юзера.

---

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

- **2026-07-24** — **★★★ РАЗВЕДКА ТОП-ЛЕСТНИЦЫ + LIVE-ВЕРДИКТ v070+P0 + ПЛАН §M (ждёт апрува юзера).**
  (1) v070+P0 live 107 игр: WR 48.6%, обе tracked-копии 705/758 (идентичные файлы, разница=шум); P0 ex_retreat
  сработал 4/107 (3 из 3 archaludon-побед — механика верна, охват ничтожен); archaludon 21%, lucario 29%; автор
  пилота prvsiyan сам на 737 → потолок гибрида ≈750, BC-потолок 900 подтверждён. (2) Разбор всех 1273 игр топ-команд
  за 07-23 (workflow, 6 sonnet-агентов): 4/5 топов играют НАШ jiatu.l-лист; все — детерминированные эвристики/BC,
  не search; топ-1 WR всего 56%; зеркало Grimmsnarl = 40-48% игр топ-бэнда, решается 5 механиками (ex-ротация до
  атаки / Adrena каждый ход / splash-double-prize / Rare Candy темп / Boss→чужой Munkidori); Handheld Fan x2 =
  зеркальный контр-тех топ-1 (75% vs 56%). (3) Band-математика: наш блокер = болото 700-899 (Lucario+Archaludon),
  эксперт той же деки держит 53-56% во всех бэндах. Пороги 07-24: бронза 851/серебро 920/золото 1088. Полный отчёт
  `docs/gold/top_ladder_recon_20260724.md`; памяти project_top5_ladder_recon_20260724 + project_v070_p0_live_verdict_20260724.
  **План §M (v072_mirror_gauntlet): база v065 + M1 зеркальный плейбук + M2 Lucario recovery (v063-ветки) +
  M3 anti-Archaludon + M4 Xerosic-тайминг + M6 band-weighted гейт → двойной сабмит после PASS. ⏳ ЖДЁТ АПРУВА.**

- **2026-07-22 (r)** — **★★★ ФОРЕНЗИК ВСЕХ 86 LIVE-ИГР v062 → v063_forensic86 ПОСТРОЕН И ГЕЙТ PASS.** По одному
  sonnet-субагенту на эпизод (workflow, схема-форсированный вывод) + ручная кросс-валидация. Найден и воспроизведён
  **критический баг**: `_promotion_guard` терял bench-слот 0 (`int(o.get("index") or -1)`, Python-фолси 0) — здоровый
  Grimmsnarl ex в слоте 0 был невидим, в актив шёл Munkidori (~5 live-игр). Ещё: промо-приоритет игнорировал энергию;
  поиск тащил мёртвые копии ex (bench-depletion лузы); нет защитного свопа на матчпойнте; застрявший небоевой актив;
  зависания на повторном select; deck-out в стенах. v063 = 8 правок (P0 фиксы + защитный своп + loop-breaker +
  deck-out guard). **Гейт N=300: 0 регрессий, v017 0.92→0.98 (z=3.6), v044 0.20→0.29, v049 0.22→0.37 (z=4.0),
  h2h vs v062 0.683, прокси-Lucario +13пп (z=4.9) — главная live-дыра. Exec-smoke linux/amd64 PASS.** Отчёт:
  `out/v062_full_download/ANALYSIS_V062_86EPS.md`. Новый скилл `/replay-analyzer` + постоянные скрипты
  (`scripts/replay_digest.py`, `extract_agent_logs.py`, `download_episodes.py`). НЕ засабмичен (ждёт «ок»).

- **2026-07-19 (b)** — **§K1+K2 выполнены.** K1: Unfair Stamp НЕ бажен (исходник движка сверен), но
  **ACE SPEC конфликт с Hero's Cape** (движок отклоняет колоду, errorType=4, эмпирически проверено на
  cg_native) — блокирует прямое добавление; взамен найдены Xerosic's Machinations (1197, не ACE SPEC,
  opp_hand>=4 порог, бьёт только руку оппонента) и Hand Trimmer (1087, Item, не ACE SPEC) как кандидаты
  для K4. K2: **исправлена ошибка в собственном раннем разборе** (higaki: Judge не "потерян по невезению",
  а сброшен нашей же логикой в ответ на opponent's Xerosic's Machinations — score_discard() не защищал
  JUDGE и не капал дубликаты ARCHALUDON_EX; daiki_H: Judge сработал штатно, просто одного сброса мало).
  Правка в `submissions/v046_judgefix/` (v046_final не тронут). Гейт N=200×3 + revalidate Crustle N=400
  (david valor eval-дисциплина: N=200 delta −2.1пп оказался шумом, на N=400 стал +1.0пп) — **0 регрессий**.
  Готово к сабмиту, ждёт «ок». Память: [[project_v046_judge_wasted_early_finding]] (исправлена),
  [[reference_forum_recon_2026_07_19_alakazam]].

- **2026-07-19 (a)** — **v046_final первые live-партии: 2/2 поражения = Alakazam Powerful Hand OHKO
  (460/420 урона сквозь 300-400 HP), Judge в обеих партиях пришёл рано (ход 2-4) но не разыгран.**
  Разбор двух реплеев (higaki ep86751133, daiki_H ep86751593) + форум-рекон (топики 712811 Unfair
  Stamp bug-report, 713608 david valor postmortem — forensic-trap warning + eval-дисциплина, 721338
  search-vs-heuristic комментарий). Записан **§K PLAN: Alakazam counter** (5 шагов K1-K5, приоритет
  на узкий баг-фикс "Judge тратится впустую" прежде смены порогов/колоды, PIMC-подход в конце очереди).
  НЕ начато выполнение — план записан, ждёт следующей сессии/подтверждения юзера на конкретные шаги.
  Память: [[project_v046_judge_wasted_early_finding]].

- **2026-07-18 (z)** — **★ РЕШЕНИЕ ЮЗЕРА: откат, ждать live v039_ft + v038.** После провала варианта C (v040 локально
  0.35, forensic исправил 1 из 3 failure modes) юзер выбрал путь C (не путать с вариантом C выше): ничего не
  запускать, дождаться live-вердиктов v039_ft (54793514 PENDING) и v038 (μ685.1, ещё копит игры). Это бесплатная
  решающая информация: если v039_ft live < v038 685 — новая дека тупик и на live; если > — направление новая дека+pool
  имеет смысл, тогда reward shaping (штраф за retreat с energized / KO-замена на E0) или pool с реальными IL/PPO
  политиками. GPU не трогать, на Kaggle не сабмитить. Ждать 150-250 live-партий (~1-2 дня). Артефакты v039_ft/v040
  сохранены локально для возобновления. Память: [[project-v040-fresh-failed]], [[project-v040-forensic-3modes]].

- **2026-07-18 (y)** — **★★ ВАРИАНТ C (fresh-start v040) ПРОВАЛ + forensic 3 failure modes.** Юзер выбрал вариант C
  (fresh-start 5M на новой деке + гибридный pool self 0.30/v009a 0.20/alakazam 0.20/grimmsnarl 0.12/archaludon 0.10/v033 0.08).
  5M шагов на vast.ai (212с), чекпойнт ppo_v040fresh_final_s5013504.pth. Локальный гейт: vs v009a=**0.348** (v038 0.700,
  v039_ft 0.423), vs v033=**0.157** (B3a регресс), vs v007=0.323, vs v036=0.367. Слабее v038 и v039_ft на 3/4 матчапов.
  К сабмиту НЕ годится. Ключевая находка: training pool_wr (v033 0.11→0.50, alakazam 0.16→0.48) НЕ→gate WR — pool использует
  деки v009a/v033, но mirror/random политики (train_ppo_v2.py поддерживает только mirror/random), а не реальные IL/PPO пилоты.
  Gate-vs-training gap сохраняется. **Forensic v040 mirror (3 игры, юзер требовал проверить проблемы v039):** из 3 failure modes
  v039 исправлен ТОЛЬКО mode 3 (MegaKanga charging, max uncharged 2 vs 3); mode 1 (retreat с energized) ТАК ЖЕ (rate 0.30≈0.33);
  mode 2 (KO-замена на E0) ХУЖЕ (0 good из 5, v039 2/3). Root cause: mirror/random pool не учит ценить энергии + нет reward-сигнала.
  Память: [[project-v040-fresh-failed]], [[project-v040-forensic-3modes]]. Рычаги: reward shaping (штраф за retreat с energized /
  KO-замена на E0) / pool с реальными IL/PPO политиками / откат к v038 / MCTS.

- **2026-07-18 (x)** — **★★★ v039_ft САБМИТ live-probe 54793514 + v038 live μ=685.1.** Юзер одобрил live-probe
  несмотря на локальный регресс (0.42 vs v009a): «давай сначала сабмит на кагл, потом вариант C». Обоснование —
  generalization gap (v038 локально 0.70, но live 685.1 = середняк, не прорыв → локальный гейт плохой прокси).
  tracked-2: {v039_ft PENDING, v038 μ685.1}; v037 вытеснен. 4 сабмита осталось. Ждать 150-250 live-партий.
  NEXT: вариант C (fresh-start 5-10M на новой деке + правка pool ближе к гейт-полю), GPU vast.ai. Память:
  [[project-v039ft-shipped-v038-live-685]].

- **2026-07-18 (w)** — **★★★★ J-fix-N2+N3 ВЫПОЛНЕНЫ: fine-tune 3M = CATASTROPHIC FORGETTING, к сабмиту НЕ годится.**
  PPO fine-tune с чекпойнта v038 на новом pool v039 + новой self-деке, 3M шагов на vast.ai RTX 5090 (~110s).
  Pool_wr в обучении росли (alakazam 0.26→0.78, entropy 0.117→0.187, mean_ret +0.59→+0.376) — НО локальный гейт
  vs реальным сабмишенам провален: v039_ft vs v009a = **0.423** [0.369,0.480] (v038 0.700 → Δ−27.7пп); vs v007 0.428;
  vs v033 0.230; vs v036 0.350. 0 exceptions. **Факторный 2×2 (WR vs v009a):** old-ckpt+old-deck=0.700 (v038),
  old-ckpt+new-deck=0.153 (no-retrain), new-ckpt+new-deck=0.423 (ft), **new-ckpt+old-deck=0.418** (эксперимент).
  new-ckpt+old ≈ new-ckpt+new (CI перекрываются) → возврат старой декы НЕ восстановил v038 → **виновато обучение/pool,
  не дека (catastrophic forgetting)**. Причина: pool v039 рассогласован с гейт-полем (pool-alakazam≠v009a/v033);
  политика переучилась на лёгких pool-оппонентах. К сабмиту НЕ годится — ухудшил бы tracked-2 (v038 уже PENDING).
  Рычаги: KL-anchor к old-ckpt / pool ближе к гейт-полю / fresh-start; либо откат ждать live v038. Память:
  [[project-v039-finetune-forgetting-2x2]].

- **2026-07-18 (v)** — **★★★ J-fix-N1 ВЫПОЛНЕН: deck-swap БЕЗ retrain = КОЛЛАПС.** Локальный гейт
  `scripts/gate_expanded.py` (cg_native, non-god-mode, N=300, seed=20260716): v039 (новая дека Crustle+MegaKanga,
  старый чекпойнт v038) vs v009a WR **0.153** [0.117,0.199] (v038 baseline 0.700 → Δ−55пп); vs v007 0.160 (Δ−50пп);
  vs v033 0.093 (Δ−34пп); vs v036 0.200 (Δ~0). 0 exceptions, ~73 хода/игру → пилот играет но проигрывает (deck-policy
  entanglement: NN обучена на card-ID старой деки, своп 31→13 энергий + смена ядра карт → неадекватные выходы).
  Вердикт: WR vs v009a 0.153 ≪ 0.55 → deck-swap без retrain НЕ ценен, **fine-tune J-fix-N2 ОБЯЗАТЕЛЕН (GPU, ждёт «ок»)**.
  Файлы: results/gates/v039_vs_*.json. Память: [[project-v039-deckswap-no-retrain-collapse]].

- **2026-07-18 (u)** — **★★ v039 ДЕКА+POOL ПРАВКИ ПРИНЯТЫ юзером, файловые правки внесены (§J-fix).**
  Юзер выбрал: для self-колоды — рекомендация #1 (Crustle+MegaKangaskhan эталон top250 rank 3, 13P/13E/34T,
  снимает сломанный баланс 31 энергия); для PPO pool — правки #4 (добавить crustle_mirror 0.12), #5 (alakazam
  0.20→0.30, обновить до живого эталона A), #6 (grimmsnarl обновить + добавить mewtwo 0.06), #7 (перевесить:
  archaludon/lucario 0.125→0.05, impidimp убран). Контекст: наша self-дека v038 = 8 покемонов/31 энергия/21
  trainer — аномалия (все топ-250 имеют 7-15 энергий); live-симптомы юзера — mulligan'ы из-за нехватки покемонов,
  deck-out от навешивания по 10 энергий. PPO pool v038 НЕ содержал живого топ-Crustle вообще (11% топа) —
  главный пробел. Артефакты: `submissions/v039_crustle_megakanga/` (копия v038 + новый deck.csv, тот же чекпойнт
  = deck-swap эксперимент), `decks/v039_pool/{crustle_megakangaskhan,alakazam_A,grimmsnarl_live,mewtwo}.csv`,
  `scratchpad/gold/plan_d/d1_train/train_ppo_v2.py` POOL_REGISTRY обновлён (старый блок сохранён как комментарий).
  PPO retrain НЕ запущен (ждёт GPU + «ок»). NEXT для другого агента: J-fix-N1 локальный гейт deck-swap vs v009a
  (бесплатно) → J-fix-N2 PPO fine-tune на новом pool+self (GPU, «ок») → J-fix-N3 packaging+live-probe («ок»).
  Память: [[project-v039-deck-and-pool-recommendations]]. План: §J-fix.
- **2026-07-18 (t)** — **★ RECON 6 Kaggle-ноутбуков → ПЛАН v039 (§J) записан.** Разобраны: myso1987
  (meta по рейтинговым бендам), ygkojima/ptcg-strategy-report-v2 (N=70K game states), naoto714 ×2
  (Mega Kangaskhan ex turn-two speed), premaananda108 (daily meta win-rate), beicicc (public experiment
  snapshot). Ключевые находки: (1) **Alakazam — монополист топа** 38-45% в 900-1100+ (myso) — подтверждает
  приоритет B3a; (2) **Marnie Grimmsnarl — топ-2** 14-15% в 1100+, НЕ был на радаре; (3) **Mega Lucario
  исчезает в 1000+** (ноуб-стомпер 500-700), **Archaludon 0% в 1100+** — объясняет потолок v036 μ831;
  (4) **hand-disruption (Unfair Stamp 1080 + Xerosic 1197) против Alakazam** — новый угол для B3a (naoto714
  локально +23.9пп), но локальный гейт vs Alakazam врёт → проверять только live-форензиком; (5) **нелинейная
  prize-diff→WR кривая с плато +3** (ygkojima) — дешёвый фикс для leaf eval/reward shaping; (6) **band-specific
  matchup inversion** (Alakazam vs Kangaskhan 61% band≥1100 vs 28.7% band 800-1099) — объясняет часть
  gate-vs-live gap, гейт надо читать по бендам; (7) BC-баги D01/D03 (turn-index rescaling, own-bench absent) —
  проверить в наших пайплайнах; (8) 4 архетип-кандидата для replay-mining (Tien Search-Alakazam 955.6, LLCC
  Crustle AttackFix 964.3, Pilkwang Archaludon 1030.6, Tomato anti-Starmie 964.1). v038 уже залит (PENDING),
  править нельзя → всё в §J для v039 ПОСЛЕ live-вердикта v038. Память:
  [[reference-kaggle-notebooks-recon-2026-07-18]], [[project-v039-recon-driven]].
- **2026-07-16 (s)** — **★ I1-3 КЛОН-ПУТЬ ЗАКРЫТ: BC-потолок подтверждён 5-й раз.** Клон топ-10 Crustle+MegaKangaskhan ex
  (懒惰的金枪鱼/__Taichicchi__) — 3 итерации фичей на 726 эпизодах официального датасета (07-14/07-15, deck-signature
  {345,756}+winner). v1 features_v009a (Crustle-спец) val_top1 0.436 / H2H 0.24 → v2 (capacity) 0.447 → **v3 features_clone
  (Kangaskhan-aware, FEATURE_DIM 61→75: is_kangaskhan, retreat_to_kangaskhan, is_hand_trimmer, is_heal_card, is_heros_cape,
  is_stadium, is_boss_orders, is_search_card, is_switch, has_kangaskhan_on_bench, my_active_is_kangaskhan, has_hand_trimmer_in_hand,
  n_kangaskhan_in_hand, …) val_top1 0.487 (стена 0.45 пробита +4пп), best_iter=83, train≈val → переобучения НЕТ, потолок = сигнал
  фичей.** Гейт anchored_pool 600 игр 0 exc: **H2H vs v009a WR 0.40 [0.348,0.458] — ЗНАЧИМО хуже чемпиона** (0.24→0.40, но не
  parity); vs v001_random 0.80 (RED-FLAG passed). Importances: type_ABILITY 18%, g_turn_norm 14%, card_target_energy_count 7%,
  **card_is_switch 4.6% (#4 — единственная НОВАЯ дек-фича в топ-15)**; Kangaskhan/hand-trimmer/heal/cape/stadium/boss identity
  НЕ вошли в топ-15 — разрежены в 726 эпизодах, ранжер учится на общих сигналах, не на distintive trainer-decisions. Это ровно
  BC-потолок на setup-деке (2 эволюции + Mega + disruption). **Вердикт юзера: закрыть клон-путь, зафиксировать урок; v032b НЕ
  сабмитится.** Даже parity = бронза (930.7), не серебро (+62μ). Фичи/модель оставлены как BC-warmstart для Плана D (§I2-1).
  Артефакты: `src/il/features_clone.py`, `models/il_crustle_clone_v3.txt`, `submissions/v032b_crustle_clone/`,
  `results/gates/v032b_gate_state.json`. Урок [[project-clone-kangaskhan-bc-ceiling]]. **NEXT: §I — I0-1 (досмотреть v030 до
  150-250 игр, пассивно) + §I2 План D (PPO+BC-warmstart на GPU, D2 in_progress) — единственный оставшийся путь к серебру/золоту.**
- **2026-07-16 (r)** — **★★ PLAN v5 (SILVER-FIRST) СОГЛАСОВАН пользователем.** Цель переформулирована: серебро
  (μ925.3, top-256/5115) — база, золото (μ1093.3, top-20) — апсайд. ⚠ ИСПРАВЛЕНО 07-16 вечером (полный лидерборд 5118):
  **μ925.3 = БРОНЗА (rank256), НЕ серебро**; корректные пороги — серебро μ993.2 (rank103), бронза μ923.4 (rank256),
  золото μ1132.4 (rank11), top-20 μ1087.4 (rank20). Наш пик 930.7 = бронза, до серебра +62.5μ. См. актуальные пороги
  в шапке §I и reference_leaderboard_2026_07_16. Полный план — **§I** (ныне архив; актуальный — **§J**). Разведка:
  свежий лидерборд (5115 команд), прямое чтение форума (`kaggle competitions topic-messages`, БЕЗ тяжёлого
  deep-research воркфлоу — тот второй раз упал на инфраструктурном лимите сессии, session-limit 12:20 MSK, решили
  не ждать/не гонять повторно, юзер попросил экономить агентов при высоком % использования — записано в
  `feedback_usage_awareness.md`). Находки: **★★официальный ежедневный датасет топ-эпизодов** организатора
  (`kaggle/pokemon-tcg-ai-battle-episodes-{date}`, форум 709160, медиана μ1125.9 на 07-15) — снимает 429-блокер
  клонирования ([[project-clone-majkel-next]] обновлён); форум 717697 «RL journey» — независимый self-play PPO
  дошёл до серебра за день на 1 GPU + рекомендация BC-warmstart перед RL (усиливает План D); форум 712621 —
  диапазон мэтчмейкинга узкий вокруг своего рейтинга (наш field-гейт vs топ-19 дек не отражает ранних живых
  оппонентов); форум 726690 — темп игр 6-10× зависит от рейтинга/неопределённости (наши v030/v007_resub мерены на
  ~68-125 игр/день, ближе к baseline 48/день, чем к оптимистичным цифрам из старой памяти). Детали —
  [[reference-official-episode-dataset]], [[reference-forum-recon-2026-07-16]]. **Пробел в логе 07-13..07-16
  (v024_cardemb_rl → v030_time_budget) не был записан построчно в реальном времени — см. `results/submissions.csv`
  как источник истины по числам; кратко: v024 card-embedding+RL live paritet (~640), v026-v027 ExIt+GBDT-value
  первые NN-агенты с реальным MCTS но первые 2 (v026/v027) молча работали как greedy-fallback (numpy/import баги,
  см. [[project-kaggle-live-safety]]), v028 нашёл и поправил import-баг но словил НОВЫЙ numpy ABI баг на
  GBDT-джойблибе, v029 фикс = pure-numpy linear value (AUC 0.68) — первый реальный MCTS live, v030 = + фикс модели
  оппонента (было `[1072]*N` заглушка, [[project-opponent-placeholder-bug]]) + тайм-бюджет 1.0с/ход — сильнейший
  локальный гейт за проект (WR 0.7475 [0.683,0.803], N=200), live-вердикт ещё не готов (47 игр на 07-16).** Старый
  вектор §H (PIMC/дек-viability/ExIt-lite) остаётся ЗАКРЫТ — §I его не отменяет, а идёт другим путём.
- **2026-07-13 (q)** — **★★ RL/ExIt ПРОГОН: PLATEAU (H2H 13→37→34% vs v009a), НО field 0.945 >> v009a 0.843 → LIVE-PROBE
  ОДОБРЕН.** Полный ExIt (5 итераций, ~9мин, kiyotah scaffold AlphaZero-style): H2H vs v009a плато 32-38% (чемпион не
  побит в Crustle-зеркале — v009a-IL специализирован), НО non-mirror здоровый (v018 100%, v017 ~50%, league 6
  архетипов + KL-anchor без коллапса §S). **Field-тест (я догнал): RL iter2 vs 19 мета-дек = 0.945 ≈ frozen
  card-embedding 0.946, ОБА >> v009a 0.843.** Гипотеза: v009a тонет live (μ733) т.к. мета анти-Crustle; универсальная
  NN-модель может дать live μ выше — узнать ТОЛЬКО live-probe. **Юзер ОДОБРИЛ сабмит NN-модели.** Упаковка
  `submissions/v024_cardemb_rl/` (RL iter2 model + kiyotah энкодеры + Crustle-дека, torch в eval есть, never-throw,
  exec-без-__file__) + presubmit-гейт запущены. После проверки — competition submit (одобрен). Артефакты:
  `scratchpad/gold/rl_full/`, `scripts/exit_full_run.py`. ИТОГ НОЧИ: BC-потолок подтверждён (card-embedding+RL плато),
  но универсальная NN может обойти тонущий Crustle на сместившейся мете — live решит. [[project-gold-target]]
- **2026-07-13 (p)** — **★★ CARD-EMBEDDING ВЕРДИКТ: PARITY, BC-потолок подтверждён (4-й раз). Ветка закрыта.** v2 обучен
  (val 52.9%→58.3%, переобучение снято через ×5 данных 54k + регуляризация). Конвертирован в играющего агента
  (`scripts/cardemb_agent.py`, kiyotah энкодеры + MyModel inference, 0 illegal/exceptions, 0.16мс/ход). **H2H vs наш
  v009a LightGBM на РАВНОЙ Crustle-деке: WR 0.497 (194-196, n=400, Wilson[0.448,0.547])** — чистый паритет, НЕ пробил
  потолок. Причины: мультиархетип размывает Crustle-сигнал; 27% решений в greedy-fallback (мультиселект вне покрытия);
  offline 58% ≠ live-сила (PokéForge подтверждён). Field 0.946 vs v009a 0.843 — но против СЛАБОГО greedy, H2H надёжнее.
  torch в competition eval подтверждён доступен (python:v163). **BC-потолок ~930-1000 подтверждён РЕАЛЬНЫМ 4-й
  независимый раз** (наш IL 930, PokéForge 802, kazuki-клон, card-embedding paritet). Инфра card-embedding построена и
  рабочая (переиспользуема). Артефакты: `scripts/cardemb_{agent,gate}.py`, `results/gates/cardemb_gate_*`,
  `scratchpad/gold/cardemb_out_v2/`. Развилка у пользователя: Crustle-only дообучение / RL поверх embedding (GPU есть,
  но self-play CPU-bound) / вернуться к дешёвому Budew-патч. Честно: BC-пути упёрлись, top-20 требует RL или TCG-эксперта.
  [[project-gold-target]] [[project-search-vs-rl-ceiling]] [[project-bc-combo-failure]]
- **2026-07-12 (o)** — **★ ЦЕЛЬ→top-20; CARD-EMBEDDING на Kaggle GPU РАЗВЁРНУТ.** Юзер: хочу золото→реалистичнее
  top-20 (μ~1079). Полное исследование (2 workflow, ~30 агентов): золото ~4%, top-20 ~10-15%, реальный max 850-1000μ.
  Публичные «сильные» агенты МЕРТВЫ live (romanrozen «LB950»=487, Alakazam-5th=409). Budew-патч переоценён (+20-40
  не +150: Budew 67% даже без Kangaskhan/Boss — сила пилота μ1226). **★ GPU ДОСТУПЕН БЕСПЛАТНО через Kaggle kernels
  push (enable_gpu) — снят блокер RL/embedding.** Юзер выбрал: досбор данных → card-embedding на GPU. СДЕЛАНО:
  (1) свежий мультиархетипный IL-датасет `data/il/train_meta_2026-07-12.npz` (275 эпизодов, 48 экспертов μ1000-1227,
  Alakazam/Grimmsnarl/Crustle/Cubchoo/Starmie); (2) kiyotah RL-sample энкодеры (per-card-id EmbeddingBag + Transformer)
  ПОДТВЕРЖДЕНО совместимы с нашими реплеями (to_observation_class); (3) card-embedding BC-пайплайн построен:
  `scripts/build_card_embedding_data.py` + `train_card_embedding.py` + датасет `data/il/cardemb/{train,val}.pt`
  (13784+2304 решений, random baseline 21%, avg 8 опций); (4) **GPU-kernel `maxsim73/ptcg-card-embedding-il-train`
  ЗАПУШЕН, статус RUNNING** — supervised BC, цель top-1 acc ≥74% vs LightGBM 68.5%. ⚠ РИСК переобучения (12M
  embedding-params vs 14k решений), диагностируем по val-кривой. Далее: дождаться GPU → если acc≥74% → конвертировать
  в submission-агента → live-probe (offline≠live, PokéForge). [[project-gold-target]]
- **2026-07-12 (n-исправлено)** — **v023 μ 756→715→658 за 13ч — но РАНО СУДИТЬ (поспешный вердикт отозван).**
  КРИТИЧНАЯ ОШИБКА В МОЁМ ПРЕДЫДУЩЕМ ДОКЛАДЕ: сравнивал v023 (53 игры, 13ч, стартовал μ600) со зрелыми v007/v009a
  (250 игр, 9 дней, устаканились). **Ре-сабмит стартует с μ600 и карабкается НЕДЕЛИ** (форум 714189) — v007 в
  возрасте 13ч тоже был ~650, не 750. μ на 53 играх = ШУМ, ещё растёт от 600. «Монотонное падение» = откат первого
  оптимистичного всплеска + болтанка малого n, НЕ доказательство «tech навредил». Данных для вердикта НЕТ. Юзер
  (справедливо) остановил откат: «откатывать не надо, ещё успеем». v007 НЕ залит (отменён на pre-submit).
  ПРАВИЛЬНОЕ сравнение: где v023 через ~150-250 игр (дни), по ЧИСЛУ игр а не часов. tracked-2 = v023+v018 пока
  слабые, но 5 недель до дедлайна — время есть. **УРОК: не измерять свежий сабмит (μ600-старт) против зрелых и не
  паниковать на раннем n — это forensic-trap наоборот.** Про сам вектор честно: большое исследование дало скромный,
  но честный вывод — Crustle-подход имеет потолок ~730-750 (PIMC не помог т.к. дека сильная; tech = паритет);
  реальный скачок требует обучения политики (RL/PPO), не тюнинга. [[project-plan-v4]] [[project-submissions]]
- **2026-07-12 (m)** — **v023 дрейф 756→715 (ранняя волатильность, ≈baseline); ExIt-lite НЕ стоит прогона.**
  v023 μ: 756.1 → **715.2** за ~15 мин (малый n, μ прыгает; 715 в диапазоне baseline v009a=724). Пока **v023 ≈
  чистый Crustle, НЕ явный плюс** — подтверждает локальный паритет (плюс был только на смещённом поле, живое поле
  возможно не так смещено на Alakazam как надеялся recon). Нужно больше игр для вердикта (μ не стабилизирован).
  **ExIt-lite smoke (F3-3): каркас ПОЧИНЕН** (4 бага: harness cg→cg_native, exit_loop feature_module хардкод
  features_iono, pilot FEATURE_NAMES, features_v009a off-by-one — падал бы на 100% данных; smoke прошёл, win-filter
  60.6%, LightGBM переобучился). **НО Step 1 жёстко юзает PIMCAgent как improvement-operator (PIMC закрыт).** Пути:
  (a) plain-pilot self-play = self-distillation (recon: не проходит учителя, DAgger плато 41%, «никогда не лучше
  учителя»); (b) PIMC-guided = воскрешать мёртвую ветку. **ВЫВОД: ExIt-lite не стоит полного прогона — оба в
  известные потолки.** Каркас сохранён рабочим. Артефакты: `scripts/exit_smoke.py`, фиксы в `src/exit/*`,
  `src/il/features_v009a.py`. [[project-search-vs-rl-ceiling]] [[project-plan-v4]]
- **2026-07-12 (l)** — **✅ v023 СУБМИТ COMPLETE, ранний live 756.1.** Self-validation прошла (0 краш, exec-no-__file__
  оправдался). Ранний μ **756.1 > baseline** (v007=750.7, v009a=724.1) — tech-версия на старте выше обоих чистых
  Crustle. ОСТОРОЖНО: ранний score по малому n, широкий CI; реальный сигнал за часы (48 игр/день). Читаю как «tech НЕ
  навредил, скорее чуть лучше» — survivorship-регрессии лёгких матчапов не утопили. v018 держит 485 (BC-Grimmsnarl
  тонет). tracked-2 станет v023 + v007. NEXT: ждать стабилизации μ (>750 успех / ~724 паритет / <700 откат); пока
  набирает — ExIt-lite (win-filter smoke, каркас готов). [[project-plan-v4]]
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
