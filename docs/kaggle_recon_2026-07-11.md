# Kaggle Recon — 2026-07-11

> Синтезирован оркестратором из workflow `wf_c05e4860-38c` (95/153 агентов успешно; 58 упали
> на session-429 субагентов). Скачано **+357 эпизодов** (32→389 top-level), **169 топик-дампов**
> в `docs/kaggle_recon/`, 83 топика с релевантным intel. Парсинг реплеев и авто-синтез 429-нулись —
> дайджест написан вручную из journal.jsonl. Сырьё: `scratchpad/kaggle_recon/_discussions_relevant.jsonl`.

## 0. TL;DR — что меняет наши решения

1. **★ Search ОБРАТНО пропорционален качеству базовой политики** (независимое подтверждение с числами):
   beam search **+11.3pp на слабой Alakazam** (10.7→22.0%), **−15.4pp на сильной Starmie** (54.0→38.6%).
   → Наш Crustle-IL **сильный (~906 live)**. Mirror PIMC 0.40→0.375 после sort-фикса — это **не только баг,
   а фундаментальное свойство**: search поверх сильного prior'а деградирует. Прямо объясняет провал arm(a).
2. **★ Crustle IL + PPO = +65 ELO над rule-based** (измерено другой командой) — **единственный
   количественно-подтверждённый путь наверх поверх нашего чемпиона**. PPO поверх BC с KL-penalty (0.01–0.09)
   к frozen BC prior. Raw policy WR всё ещё низкий без search, но +65 ELO на лидерборде — реальный сигнал.
3. **★ Card-embedding transformer с per-card identity** пробил feature-потолок BC: 9 методов
   (ISMCTS, AlphaZero-lite, MLP self-play, FSP, Demo-RL) плато на ~68.5% т.к. кодировали только type/damage,
   НЕ «какая карта». Per-card one-hot over vocab → 73–74% mirror. Наш IL мог упираться в **representation**, не capacity.
4. **Мета сместилась опять:** Starmie (keidroid Mega Starmie ex #1, 67% ladder), Archaludon-ось, Alakazam >1100
   (e-toppo — дека РАБОТАЕТ, наш BC был проблемой), Festival Lead >1100. Grimmsnarl/Cubchoo (наш recon 07-03)
   уже частично вытеснены. Lucario = 42% поля у keidroid, mirror-lever.
5. **Offline-гейты НЕ предсказывают live — снова, много раз:** Dudunsparce rebuild 369 несмотря на passing
   offline-гейты; hill-climbing ниже noise floor (N=2000 → 1.37pp чистый шум при 0.5pp пороге); forensic
   loss-analysis = ловушка (−7.6pp, p=0.003, survivorship bias). **Прямо валидирует нашу H2-калибровку
   как ранжировщик, не прогноз, и требует N≥1200 + ≥0.5pp пороги.**

## 1. Search / PIMC / RL — прямо релевантно текущему вектору

### Против нашего PIMC (усиливает сомнение)
- **Beam value ∝ 1/quality базы:** +11.3pp weak Alakazam, −15.4pp strong Starmie. Crude leaf eval
  (prize_diff×10000 + energy×120) перебивает нюансный tempo. Latency ~8.6ms/decision (бюджет НЕ бутылочное горло).
- **One-ply engine search: 30.0% mirror WR** — горизонт слишком короткий, хуже лучшей имитации.
- **Leaf-eval anti-pattern:** «убить пре-эво сейчас» окупается через 2-3 хода; prize-only leaf делает это
  невидимым. Наш S4b (rollout обеих сторон = Crustle-IL) мог иметь этот blind spot.
- **Clairvoyance bug:** порядок колоды фиксируется ДО Psychic Draw → MCTS «знает» что вытянет →
  систематическое завышение EV. Раздельные rollout + re-determinize на root это частично лечат.
- **Determinized-MCTS/ISMCTS** явно назван рекомендуемым паттерном; «card=action, policy prior для move
  ordering, value net или rollout для eval». Наш подход в русле, но…
- **Search API семантика (измерено сообществом):** draw из того же SearchId **ДЕТЕРМИНИРОВАН**
  (Lunar Cycle ×8 = те же 3 карты); re-determinize только на root (SearchBegin с новым порядком) даёт
  разные дро. **Подтверждает: re-determinize работает ТОЛЬКО на root-уровне** — наш H3-2 дизайн верен.
- `shuffle-then-draw` (Lillie's Determination) — НЕ детерминирован при re-step; `draw-top` — детерминирован.

### За RL/PPO поверх BC (альтернативный вектор)
- **Crustle IL + PPO = +65 ELO** над rule-based + выше offline WR. PPO: 20 итераций self-play
  (50% self-play + 50% vs heuristic), KL-penalty 0.01–0.09 к frozen BC prior. Стабильно, но raw WR ~25% vs
  heuristic без search (то есть +65 ELO приходит от ладдер-динамики, не от dominance).
- **RL plateau = measurement limit, не algorithmic:** NN value AUC 0.61 vs 10-фичный ЛИНЕЙНЫЙ public-board
  AUC 0.818; hidden hand +0.007. Bottleneck = sparse terminal-only reward + evaluator ниже своего noise floor
  (±16pp при N=120). Скрытая инфа НЕ бутылочное горло.
- **Self-play value net:** mirror 42→54.5% (Wilson lo 47.6%), но **non-mirror generalization ПРОВАЛ**
  (Wilson lo 0.16–0.18). Mirror-выигрыши не переносятся.
- **DAgger плато ~41% H2H**; iter 4 covariate-shift inversion (99% clone accuracy = 28% H2H). Чистая имитация
  не проходит эвристик-учителя.
- **AlphaZero-style RL «никогда не станет лучше учителя»** (4 голоса).
- **Kaggle worker: 1s timeout → ~10 rollouts/turn для Python-MCTS** → нужен PUCT (policy prior) чтобы
  сфокусировать 10 rollouts на top-2/3. Наш engine search API ~3600/ход = **×360** — подтверждает ценность
  engine-search над кастомным Python MCTS.

### Общие техники (borrow)
- **CRN (Common Random Numbers):** перехват native shuffle → identical policies = identical trajectories.
  Variance reduction **4.88x для near-clones, 1.18x для dissimilar**. Мерить per-comparison. → наш paired PIMC/A-B eval.
- **Hill-climbing noise floor:** identical config N=2000 = 1.37pp чистый шум. Требование: **N≥1200 + ≥0.5pp
  порог** иначе читаем шум как сигнал.
- **Forensic loss-analysis = ТРАП:** LLM-guards из 40 losing games → −7.6pp (p=0.003) survivorship bias.
  **Прямо предупреждает против нашего планового ExIt-lite / loss-analysis** — paired-A/B любой «фикс».
- **Deck repair = самый дешёвый рычаг:** basic energy 33→13 + search → WR 0.6%→9.2% при неизменной политике.
  **Структура колоды до сложности политики.**

## 2. Мета (по состоянию на ~2026-06-26…07-09)

- **Starmie ось:** keidroid LB#1 = Mega Starmie ex (330 HP, не 250), 78W/38L 67.2% ladder. Starmie vs Crustle
  только 39% (слаб), vs Trevenant 82.9%. **Lucario = 42% их поля** (mirror-lever, Starmie 29% WR при 42% весе).
- **Alakazam >1100** (e-toppo, 2x gold medalist) — **дека РАБОТАЕТ на высоком уровне; наш BC/IL был проблемой**
  (v016/v017 FAIL гейт 43%). Комбо-дека (Powerful Hand = 20 counters/карту в руке), уязвима к hand-disruption.
  Проигрывает Iono Lightning 8-12 (40%).
- **Archaludon** metal tempo = отдельная ось, rank-1 в бакете к 06-28; контрится Cornerstone Mask Ogerpon ex
  (Ability-иммунитет полностью стенит).
- **Grimmsnarl ex** подтверждён как реальный top-cut (Marnie's Grimmsnarl 320HP Stage2 Darkness) — корроборирует
  наш pivot-target. Но мета уже шире.
- **Festival Lead (おまつりおんど) >1100** — новая, не в нашем recon.
- **Baxcalibur defensive** (200+ dmg → 0) достиг rank 60 — non-RL, non-meta дека в mid-LB.
- **Crustle (наш чемпион):** Mysterious Rock Inn = блок ВСЕХ атак от opponent's ex; саппорт Cornerstone Mask
  Ogerpon 117 + Psyduck 858 Damp + Team Rocket's Factory 1257. **Iono жёстко контрит: Biohack Crustle 0-10 vs
  Sample Iono.** Объясняет ~930 μ через anti-ex lockdown, но и уязвимость к Iono/disruption.
- **Публичный baseline-матрикс** (11 агентов, 550 игр): Kokinn Lucario 63%, Biohack Crustle 63%, Iono 44%.
- **Топ-команды (06-26):** Yushin Ito #1, keidroid, Mogja J, Shun, tomatomato, Kadoraba, aidy, TG_Aetheryx,
  Jaga, Yasuo. Поле **~3550 команд** (07-09). masatomatsui = известный search-based автор (recon-таргет).
- **1091 = скор эталонных эвристик** на LB. Rank-2 = 3-режимная rule-based (Normal/Matchup/Lethal).
  Deck-specific+matchup-specific handcrafted дошёл до #12.

## 3. Движок / API / данные (техдолг и грабли)

- **Memory leak в libcg.so** подтверждён независимо — перезапуск worker каждые 25-50 матчей при self-play.
  (Мы знали threading=SIGSEGV, утечку — нет.)
- **600s TOTAL на игру, БЕЗ per-turn increment** (точное число). **Init/выбор колоды ВХОДИТ в бюджет.**
  ~1.6 vCPU. Нужен wall-clock budgeter (у нас есть — bank-manager).
- **Deck-search blind:** `observation.select.option` при deck search = только `{type:3, area:1, index:N,
  playerIndex}` БЕЗ cardId — агент «слепой» при поиске по колоде.
- **min/maxCount semantics:** индексы уникальны, len ∈ [minCount, maxCount], иначе err=4. Порядок значим ТОЛЬКО
  в SKILL_ORDER=34 и FIRST_EFFECT=44; везде — set.
- **Off-by-one (717141) количественно:** ~20% same-step (obs,action) пар имеют action вне range при
  minCount==maxCount==1; 35% mismatch при naive same-step pairing. **← ПОДТВЕРЖДАЕТ наш H1-аудит:
  naive same-step битый, но наш parse.py использует `steps[i].obs ↔ steps[i+1].action` = чистый.**
- **Observation schema дрейфует между шагами:** `current` vs `observation.current` vs `obs.current`.
  **Наш `src/replays/parse.py` обрабатывает только `observation` — steps с `obs` молча пропускаются
  (потенциальный data-gap).** ← actionable robustness-фикс если вернёмся к IL.
- **EN card IDs** используются движком (verified: наш Crustle 344/345 → μ930). EN_Card_Data.csv **НЕПОЛНЫЙ**
  vs реальный пул (Baxcalibur/SV7 отсутствуют). JP vs EN — разные expansion-коды, IDs пересекаются численно
  но мапятся на РАЗНЫЕ карты (ID 158 = Drednaw EN vs Baxcalibur JP).
- **Submission size limit: 197.7 MiB.** Новый libcg.so + PIMC-ассеты должны влезть.
- **Native cg НЕ на pip kaggle-environments** (другой card-ID universe) — search-сабмиты обязаны бандлить
  новый native libcg.so (подтверждает project_native_engine).
- **NO INGRESS/EGRESS** — eval-песочница сетево-изолирована, никаких внешних API на inference (исключает LLM-in-loop).
- **Step-limit увеличен (716045, 06-30):** циклящий агент проигрывает по TIMEOUT.
- **16 турнирных top-cut дек в competition card-id формате** (yu0307/deck-csv) — drop-in deck.csv БЕЗ конвертации
  → готовые opponent models для anchored pool/gauntlet.
- **5 официальных starter-ноутбуков** (kiyotah): Iono, Dragapult, Mega Abomasnow, Mega Lucario, **RL & MCTS
  sample code**.

## 4. Governance / правила
- Re-submit НЕ наследует рейтинг, старт μ600 (подтверждает 714189). Активны 2 ПОСЛЕДНИХ сабмита.
- Single-deck = requirement (host: metagame-awareness часть вызова, но general-purpose НЕ ожидается —
  валидирует наш deck-specific подход). Enforcement multi-deck для Simulation = «discussing», не подтверждён.
- Winners грант license/assignment победных сабмишенов. Шаринг только на Kaggle forums (X/non-public = violation).
- +0 rating за win при большой разнице рейтингов — норма. Elo fixed-point:
  `settled ≈ mean_opp_rating + 400·log10(p/(1-p))` — объясняет почему проигрышный-рекорд-агент оседает высоко.

## 5. Action items (для наших задач)
1. **PIMC re-gate (H3-4-re):** ожидание mirror<0.55 ТЕПЕРЬ теоретически обосновано (search ∝ 1/quality). Если
   arm(b) FIELD-дельта после sort-фикса тоже ≤0 — это **не баг, а фундаментальный потолок** → честный KILL.
   Если FIELD-дельта стала >0 (sort-баг вредил vs реальных оппонентов) — PIMC жив как FIELD-инструмент даже при
   слабом mirror. Гейт: N≥1200 для значимости (noise floor), ≥0.5pp пороги.
2. **Новый вектор-кандидат: Crustle IL + PPO (+65 ELO).** Единственный измеренный путь наверх поверх чемпиона.
   PPO с KL-penalty к frozen BC prior. Требует reward-сигнала (терминальный WR) + self-play инфры (у нас есть
   native engine, но memory-leak → process-isolation каждые 25-50 матчей).
3. **Card-embedding для IL:** если вернёмся к BC — per-card identity one-hot пробивает feature-потолок 68.5→74%.
   Наш `features_*.py` кодирует type/damage — проверить, есть ли per-card identity.
4. **parse.py robustness:** обработать `obs`/`current` ключи (не только `observation`) если вернёмся к IL-данным.
5. **Готовые opponent models:** 16 top-cut дек (yu0307/deck-csv) → расширить anchored pool реальным полем,
   не только своими агентами (наша давняя слабость — пул скошен к Crustle-семейству).
6. **Deck-repair рычаг:** дешевле политики. Если строим новую деку — energy count + search важнее пилота.

## 6. Незакрытое (429-провалы, добрать позже)
- **439 эпизодов ещё не скачано** (796 missing − 357 done); приоритет был свежие v018/v007.
- **~40 discussion-топиков не выкачаны** (709xxx батч 429-нулся) — но это в основном старые июньские треды.
- **Парсинг 389 реплеев в summary НЕ выполнен** (12 parse-батчей 429). Реплеи на диске — можно распарсить
  локальным python-скриптом (`src/replays/parse.py`) БЕЗ субагентов, если понадобится episode-level анализ
  наших live-игр (кого бьём/кому проигрываем по архетипам).
- Топики для добора: 711644 (RL/PPO/MCTS опыт), 713608 (What Ceilinged), 715117 (место 60 без RL),
  709152 (memory leak), 711329 (API extension для search), 712226/712811/714030/714444 (баги движка).
